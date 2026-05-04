//! Local Search: entity-anchored, token-budgeted context packer (Edge et al. 2024).
//!
//! Priority tiers, packed in order until the budget is exhausted:
//!   (a) Entity descriptions    — define the answer, cheapest tokens, dropped last
//!   (b) Relationship descriptions — connections among the seed entities
//!   (c) Source-chunk text      — chunks where the seed entities were mentioned
//!   (d) Covering community context — community summaries the seeds belong to
//!
//! Overflow drops from the lowest-priority tier first; tiers (b–d) are skipped
//! whole if even one item would exceed the remaining budget. Token counting
//! uses a `chars / 4` heuristic — accurate enough for budget gating, swappable
//! for `tiktoken-rs` once it lands on the workspace (see PR #126).

use crate::{
    core::{ChunkId, Entity, EntityId, KnowledgeGraph, Relationship},
    Result,
};
use std::collections::HashSet;

/// Token budget tiers, in pack-priority order. Lower index = higher priority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalContextTier {
    /// Entity descriptions (highest priority).
    Entities,
    /// Relationship descriptions among the seed entities.
    Relationships,
    /// Source chunks where the seed entities are mentioned.
    SourceChunks,
    /// Covering community summaries.
    Community,
}

impl LocalContextTier {
    /// Human-readable label for this tier (for diagnostics / display).
    pub fn label(self) -> &'static str {
        match self {
            LocalContextTier::Entities => "Entities",
            LocalContextTier::Relationships => "Relationships",
            LocalContextTier::SourceChunks => "Sources",
            LocalContextTier::Community => "Communities",
        }
    }
}

/// Configuration for a `LocalSearch` invocation.
#[derive(Debug, Clone)]
pub struct LocalSearchConfig {
    /// Token budget for the assembled context.
    pub budget: usize,
    /// Maximum number of seed entities harvested from the query.
    pub top_k_entities: usize,
    /// Whether to expand to depth-1 neighbours when collecting relationships.
    pub include_neighbors: bool,
}

impl Default for LocalSearchConfig {
    fn default() -> Self {
        Self {
            budget: 2048,
            top_k_entities: 8,
            include_neighbors: true,
        }
    }
}

/// Packed local-search context, organized by priority tier.
///
/// Each `Vec<String>` holds the items that fit in the budget for that tier.
/// `dropped_tier` records the first tier that overflowed (if any) so callers
/// can surface budget-exhaustion to users.
#[derive(Debug, Clone, Default)]
pub struct LocalContext {
    /// Names of the seed entities the context was anchored on.
    pub seed_entities: Vec<String>,
    /// Tier (a): entity descriptions.
    pub entity_descriptions: Vec<String>,
    /// Tier (b): relationship descriptions.
    pub relationship_descriptions: Vec<String>,
    /// Tier (c): source-chunk text.
    pub source_chunks: Vec<String>,
    /// Tier (d): covering community summaries.
    pub community_context: Vec<String>,
    /// Total tokens (estimated) consumed by the assembled context.
    pub total_tokens: usize,
    /// Token budget the context was packed against.
    pub budget: usize,
    /// First tier that hit the budget cap, if any.
    pub dropped_tier: Option<LocalContextTier>,
}

impl LocalContext {
    /// True when no entity matched the query.
    pub fn is_empty(&self) -> bool {
        self.entity_descriptions.is_empty()
            && self.relationship_descriptions.is_empty()
            && self.source_chunks.is_empty()
            && self.community_context.is_empty()
    }

    /// Render the context as a single prompt-ready string with section headers.
    pub fn to_prompt(&self) -> String {
        let mut out = String::new();
        if !self.entity_descriptions.is_empty() {
            out.push_str("## Entities\n");
            for d in &self.entity_descriptions {
                out.push_str("- ");
                out.push_str(d);
                out.push('\n');
            }
        }
        if !self.relationship_descriptions.is_empty() {
            out.push_str("\n## Relationships\n");
            for d in &self.relationship_descriptions {
                out.push_str("- ");
                out.push_str(d);
                out.push('\n');
            }
        }
        if !self.source_chunks.is_empty() {
            out.push_str("\n## Sources\n");
            for c in &self.source_chunks {
                out.push_str(c);
                out.push_str("\n\n");
            }
        }
        if !self.community_context.is_empty() {
            out.push_str("\n## Communities\n");
            for c in &self.community_context {
                out.push_str("- ");
                out.push_str(c);
                out.push('\n');
            }
        }
        out
    }
}

/// Estimate tokens for a string using a `chars / 4` heuristic.
///
/// `tiktoken-rs` would be more accurate; the heuristic is within ~15% for
/// English prose and is good enough for budget gating. Tests rely on this
/// function being deterministic and side-effect-free.
pub fn estimate_tokens(text: &str) -> usize {
    text.chars().count().div_ceil(4)
}

/// Local Search engine: produces a packed `LocalContext` for a query.
///
/// `LocalSearch` does not need ownership of the retrieval system or async
/// embeddings — it operates directly on the knowledge graph using
/// case-insensitive entity matching for seeds. This keeps it usable from sync
/// contexts (CLI, FFI) and side-effect-free.
pub struct LocalSearch<'a> {
    graph: &'a KnowledgeGraph,
    config: LocalSearchConfig,
}

impl<'a> LocalSearch<'a> {
    /// Build a new `LocalSearch` over `graph` with the given config.
    pub fn new(graph: &'a KnowledgeGraph, config: LocalSearchConfig) -> Self {
        Self { graph, config }
    }

    /// Build with default config.
    pub fn with_default_config(graph: &'a KnowledgeGraph) -> Self {
        Self::new(graph, LocalSearchConfig::default())
    }

    /// Run local search and return a packed context fitted to `budget` tokens.
    ///
    /// `budget` overrides the config budget for this call (so a single
    /// `LocalSearch` instance can serve multiple budgets). Returns
    /// `LocalContext::default()` when no entity in the graph matches the query.
    pub fn search(&self, query: &str, budget: usize) -> Result<LocalContext> {
        let mut ctx = LocalContext {
            budget,
            ..Default::default()
        };

        let seeds = self.find_seed_entities(query);
        if seeds.is_empty() {
            return Ok(ctx);
        }
        ctx.seed_entities = seeds.iter().map(|e| e.name.clone()).collect();

        let mut remaining = budget;
        let mut dropped: Option<LocalContextTier> = None;

        // Tier (a): entity descriptions
        for entity in &seeds {
            let desc = format_entity_description(entity);
            let cost = estimate_tokens(&desc);
            if cost > remaining {
                dropped.get_or_insert(LocalContextTier::Entities);
                break;
            }
            remaining -= cost;
            ctx.entity_descriptions.push(desc);
        }

        // Tier (b): relationship descriptions among seeds + their direct
        // neighbours (depth-1).
        if dropped.is_none() {
            let rels = self.collect_relationships(&seeds);
            for rel_desc in rels {
                let cost = estimate_tokens(&rel_desc);
                if cost > remaining {
                    dropped.get_or_insert(LocalContextTier::Relationships);
                    break;
                }
                remaining -= cost;
                ctx.relationship_descriptions.push(rel_desc);
            }
        }

        // Tier (c): source chunks (deduped) for the seed entities
        if dropped.is_none() {
            let chunks = self.collect_source_chunks(&seeds);
            for chunk_text in chunks {
                let cost = estimate_tokens(&chunk_text);
                if cost > remaining {
                    dropped.get_or_insert(LocalContextTier::SourceChunks);
                    break;
                }
                remaining -= cost;
                ctx.source_chunks.push(chunk_text);
            }
        }

        // Tier (d): covering community context (placeholder until community
        // summaries are wired through #93/#128 — for now the tier is silent).
        // Intentionally left empty; the test suite asserts (a–c) priority order.

        ctx.total_tokens = budget.saturating_sub(remaining);
        ctx.dropped_tier = dropped;
        Ok(ctx)
    }

    /// Case-insensitive substring entity match. Mirrors `analyze_query` in
    /// `retrieval/mod.rs` so seeds line up with the existing classifier.
    fn find_seed_entities(&self, query: &str) -> Vec<Entity> {
        let q_lower = query.to_lowercase();
        let words: Vec<&str> = q_lower.split_whitespace().collect();

        let mut seeds: Vec<Entity> = Vec::new();
        let mut seen: HashSet<EntityId> = HashSet::new();
        for entity in self.graph.entities() {
            let name_lower = entity.name.to_lowercase();
            // Match if any whole word in the query overlaps the entity name.
            // Avoid trivial 1- or 2-char overlaps which create noisy seeds.
            let hit = words
                .iter()
                .any(|&w| w.len() >= 3 && (name_lower.contains(w) || w.contains(&name_lower)));
            if hit && !seen.contains(&entity.id) {
                seen.insert(entity.id.clone());
                seeds.push(entity.clone());
            }
            if seeds.len() >= self.config.top_k_entities {
                break;
            }
        }
        seeds
    }

    /// Build relationship descriptions for edges incident to any seed entity.
    ///
    /// Uses `get_incident_relationships`, which walks both outgoing *and*
    /// incoming edges so seeds that are only edge targets still contribute.
    fn collect_relationships(&self, seeds: &[Entity]) -> Vec<String> {
        let seed_ids: HashSet<EntityId> = seeds.iter().map(|e| e.id.clone()).collect();
        let mut emitted: HashSet<(EntityId, EntityId, String)> = HashSet::new();
        let mut out = Vec::new();

        for seed in seeds {
            for (other, rel) in self.graph.get_incident_relationships(&seed.id) {
                let key = (
                    rel.source.clone(),
                    rel.target.clone(),
                    rel.relation_type.clone(),
                );
                if emitted.contains(&key) {
                    continue;
                }
                // Prefer relationships among seeds; if `include_neighbors` is
                // off, skip edges that leave the seed set.
                if !self.config.include_neighbors && !seed_ids.contains(&other.id) {
                    continue;
                }
                emitted.insert(key);
                // Render with the canonical (source -> target) direction
                // from the relationship itself, regardless of which endpoint
                // we found it through.
                let source_entity = self.graph.get_entity(&rel.source).unwrap_or(seed);
                let target_entity = self.graph.get_entity(&rel.target).unwrap_or(other);
                out.push(format_relationship(source_entity, target_entity, rel));
            }
        }
        out
    }

    /// Collect deduped source-chunk content for the seed entities.
    fn collect_source_chunks(&self, seeds: &[Entity]) -> Vec<String> {
        let mut seen: HashSet<ChunkId> = HashSet::new();
        let mut out = Vec::new();
        for seed in seeds {
            for mention in &seed.mentions {
                if seen.contains(&mention.chunk_id) {
                    continue;
                }
                if let Some(chunk) = self.graph.get_chunk(&mention.chunk_id) {
                    seen.insert(mention.chunk_id.clone());
                    out.push(format!("[{}] {}: {}", chunk.id, seed.name, chunk.content));
                }
            }
        }
        out
    }
}

fn format_entity_description(entity: &Entity) -> String {
    format!("{} ({})", entity.name, entity.entity_type)
}

fn format_relationship(source: &Entity, target: &Entity, rel: &Relationship) -> String {
    format!(
        "{} -[{}]-> {} (confidence {:.2})",
        source.name, rel.relation_type, target.name, rel.confidence
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{
        ChunkId, DocumentId, Entity, EntityId, EntityMention, KnowledgeGraph, Relationship,
        TextChunk,
    };

    fn make_chunk(id: &str, doc: &str, content: &str) -> TextChunk {
        TextChunk::new(
            ChunkId::new(id.to_string()),
            DocumentId::new(doc.to_string()),
            content.to_string(),
            0,
            content.len(),
        )
    }

    fn make_mention(chunk_id: &str) -> EntityMention {
        EntityMention {
            chunk_id: ChunkId::new(chunk_id.to_string()),
            start_offset: 0,
            end_offset: 1,
            confidence: 1.0,
        }
    }

    fn fixture_graph() -> KnowledgeGraph {
        let mut g = KnowledgeGraph::new();

        g.add_chunk(make_chunk(
            "c1",
            "doc1",
            "Alice met Bob in Wonderland to discuss algorithms.",
        ))
        .unwrap();
        g.add_chunk(make_chunk(
            "c2",
            "doc1",
            "Bob mentored Carol on graph theory.",
        ))
        .unwrap();
        g.add_chunk(make_chunk(
            "c3",
            "doc1",
            "Unrelated chunk about weather patterns and clouds.",
        ))
        .unwrap();

        let alice = Entity::new(
            EntityId::new("alice".into()),
            "Alice".into(),
            "PERSON".into(),
            0.95,
        )
        .with_mentions(vec![make_mention("c1")]);
        let bob = Entity::new(
            EntityId::new("bob".into()),
            "Bob".into(),
            "PERSON".into(),
            0.95,
        )
        .with_mentions(vec![make_mention("c1"), make_mention("c2")]);
        let carol = Entity::new(
            EntityId::new("carol".into()),
            "Carol".into(),
            "PERSON".into(),
            0.95,
        )
        .with_mentions(vec![make_mention("c2")]);

        g.add_entity(alice).unwrap();
        g.add_entity(bob).unwrap();
        g.add_entity(carol).unwrap();

        g.add_relationship(
            Relationship::new(
                EntityId::new("alice".into()),
                EntityId::new("bob".into()),
                "MET".into(),
                0.9,
            )
            .with_context(vec![ChunkId::new("c1".into())]),
        )
        .unwrap();
        g.add_relationship(Relationship::new(
            EntityId::new("bob".into()),
            EntityId::new("carol".into()),
            "MENTORED".into(),
            0.85,
        ))
        .unwrap();

        g
    }

    /// Local search must return entity-anchored context, not unrelated vector hits.
    #[test]
    fn local_search_returns_only_entity_anchored_context() {
        let g = fixture_graph();
        let ls = LocalSearch::with_default_config(&g);
        let ctx = ls.search("Tell me about Alice", 4096).unwrap();

        assert!(ctx.entity_descriptions.iter().any(|d| d.contains("Alice")));
        // Source chunks should come from chunks Alice was mentioned in (c1)
        assert!(ctx.source_chunks.iter().any(|s| s.contains("Wonderland")));
        // The unrelated weather chunk must NOT appear.
        assert!(
            !ctx.source_chunks.iter().any(|s| s.contains("weather")),
            "vector-only chunks should not bleed into local search"
        );
    }

    /// With a budget that fits only the entity tier, no other tier should be packed.
    #[test]
    fn local_search_packs_priority_tiers_in_order() {
        let g = fixture_graph();
        let ls = LocalSearch::with_default_config(&g);

        // First measure exact entity-tier cost to compute a tight budget.
        let alice_desc = format!("Alice (PERSON)");
        let bob_desc = format!("Bob (PERSON)");
        let only_entities = estimate_tokens(&alice_desc) + estimate_tokens(&bob_desc);

        let ctx = ls.search("Alice and Bob", only_entities).unwrap();
        assert_eq!(ctx.entity_descriptions.len(), 2, "both seeds should fit");
        assert!(ctx.relationship_descriptions.is_empty());
        assert!(ctx.source_chunks.is_empty());
        assert_eq!(ctx.dropped_tier, Some(LocalContextTier::Relationships));
    }

    /// With incrementally larger budgets, lower-priority tiers fill in order.
    #[test]
    fn local_search_drops_lowest_tier_first() {
        let g = fixture_graph();
        let ls = LocalSearch::with_default_config(&g);

        // Tiny budget -> only entities, with overflow noted.
        let tiny = ls.search("Alice and Bob", 8).unwrap();
        assert!(!tiny.entity_descriptions.is_empty());
        assert!(tiny.source_chunks.is_empty());

        // Generous budget -> all of (a)+(b)+(c) populated.
        let big = ls.search("Alice and Bob", 4096).unwrap();
        assert!(!big.entity_descriptions.is_empty());
        assert!(
            !big.relationship_descriptions.is_empty(),
            "relationships must populate when budget allows"
        );
        assert!(
            !big.source_chunks.is_empty(),
            "source chunks must populate when budget allows"
        );
    }

    /// Total estimated tokens of the assembled context must not exceed the budget.
    #[test]
    fn local_search_respects_token_budget() {
        let g = fixture_graph();
        let ls = LocalSearch::with_default_config(&g);

        for budget in [16, 32, 64, 128, 512] {
            let ctx = ls.search("Alice Bob Carol", budget).unwrap();
            assert!(
                ctx.total_tokens <= budget,
                "consumed {} > budget {}",
                ctx.total_tokens,
                budget
            );
        }
    }

    /// A query with no matching entities must return an empty context, never
    /// fall through to vector-similarity-only chunks.
    #[test]
    fn local_search_with_no_matching_entities_returns_empty_context() {
        let g = fixture_graph();
        let ls = LocalSearch::with_default_config(&g);
        let ctx = ls.search("xylophones perambulate quietly", 4096).unwrap();

        assert!(ctx.is_empty());
        assert!(ctx.seed_entities.is_empty());
        assert_eq!(ctx.total_tokens, 0);
    }

    /// `LocalContextTier::label` covers each tier without panicking.
    #[test]
    fn tier_labels_are_stable() {
        assert_eq!(LocalContextTier::Entities.label(), "Entities");
        assert_eq!(LocalContextTier::Relationships.label(), "Relationships");
        assert_eq!(LocalContextTier::SourceChunks.label(), "Sources");
        assert_eq!(LocalContextTier::Community.label(), "Communities");
    }

    /// Seeds that are only the *target* of a relationship must still see that
    /// edge surface in tier (b). Regression test for the directed-graph
    /// `get_neighbors` bug fixed by switching to `get_incident_relationships`.
    #[test]
    fn local_search_includes_incoming_relationships_for_seed() {
        let mut g = KnowledgeGraph::new();
        g.add_chunk(make_chunk("c1", "doc1", "Bob and Carol talked."))
            .unwrap();

        let bob = Entity::new(
            EntityId::new("bob".into()),
            "Bob".into(),
            "PERSON".into(),
            0.95,
        )
        .with_mentions(vec![make_mention("c1")]);
        let carol = Entity::new(
            EntityId::new("carol".into()),
            "Carol".into(),
            "PERSON".into(),
            0.95,
        )
        .with_mentions(vec![make_mention("c1")]);
        let dave = Entity::new(
            EntityId::new("dave".into()),
            "Dave".into(),
            "PERSON".into(),
            0.95,
        )
        .with_mentions(vec![make_mention("c1")]);

        g.add_entity(bob).unwrap();
        g.add_entity(carol).unwrap();
        g.add_entity(dave).unwrap();

        // Carol is the *source* of one edge (outgoing) ...
        g.add_relationship(Relationship::new(
            EntityId::new("carol".into()),
            EntityId::new("dave".into()),
            "MENTORED".into(),
            0.85,
        ))
        .unwrap();
        // ... and the *target* of another (incoming).
        g.add_relationship(Relationship::new(
            EntityId::new("bob".into()),
            EntityId::new("carol".into()),
            "MET".into(),
            0.9,
        ))
        .unwrap();

        // Query mentions only Carol, so Carol is the sole seed.
        let ls = LocalSearch::new(
            &g,
            LocalSearchConfig {
                budget: 4096,
                top_k_entities: 1,
                include_neighbors: true,
            },
        );
        let ctx = ls.search("Tell me about Carol", 4096).unwrap();

        assert_eq!(ctx.seed_entities, vec!["Carol".to_string()]);
        // Both edges are incident to Carol; both must appear.
        assert!(
            ctx.relationship_descriptions
                .iter()
                .any(|d| d.contains("Bob") && d.contains("MET") && d.contains("Carol")),
            "incoming edge Bob-MET->Carol must appear, got: {:?}",
            ctx.relationship_descriptions
        );
        assert!(
            ctx.relationship_descriptions
                .iter()
                .any(|d| d.contains("Carol") && d.contains("MENTORED") && d.contains("Dave")),
            "outgoing edge Carol-MENTORED->Dave must appear, got: {:?}",
            ctx.relationship_descriptions
        );
    }
}
