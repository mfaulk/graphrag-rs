//! Element-summary collapse for entities and relationships.
//!
//! Implements Edge et al. 2024 §2.2: when the same element (entity or
//! relationship) is described in multiple chunks, the LLM synthesises a
//! single coherent description from the per-instance descriptions. Below a
//! cost-guard threshold the descriptions are concatenated locally to avoid
//! unnecessary LLM calls.
//!
//! # Status: Wired into `build_graph`
//!
//! `GraphRAG::build_graph` invokes [`collapse_all`] after entity/relationship
//! extraction and before persistence whenever the LLM extraction paths run
//! (single-pass or gleaning) and `entities.element_summary.enabled` is set.
//! The synthesised description is written back onto
//! `core::Entity::description` and `core::Relationship::description`, so
//! downstream community-report and global-search consumers see one coherent
//! description per element.
//!
//! The collapse helpers ([`collapse_descriptions`], [`collapse_all`],
//! [`collapse_entity_descriptions`], [`collapse_relationship_descriptions`])
//! remain a stable public API for callers operating on their own description
//! storage.

use std::collections::HashMap;

#[cfg(test)]
use async_trait::async_trait;

use crate::config::ElementSummaryConfig;
use crate::core::backend::{ChatBackend, ChatParams};
use crate::core::Result;

/// Prompt template for synthesizing a single description from multiple
/// instance descriptions of the same element.
pub const ELEMENT_SUMMARY_PROMPT: &str = r#"You are a helpful assistant that summarizes information about an entity or relationship.

Below are several descriptions of the same {kind} "{name}" extracted from different parts of a document. Some descriptions may overlap, contradict each other, or capture different aspects. Synthesize a single coherent description that:
- Resolves contradictions sensibly
- Preserves all distinct factual information
- Reads as one fluent paragraph
- Stays in third person and avoids meta-commentary about the synthesis itself

Descriptions:
{descriptions}

Synthesized description:"#;

/// Render a list of descriptions into a numbered block for the prompt.
fn format_descriptions(descriptions: &[String]) -> String {
    descriptions
        .iter()
        .enumerate()
        .map(|(i, d)| format!("{}. {}", i + 1, d.trim()))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Build the LLM prompt for collapsing multiple descriptions of one element.
///
/// `kind` is a short label like "entity" or "relationship", and `name` is a
/// human-readable identifier such as the entity name or `source -> target`.
pub fn build_summary_prompt(kind: &str, name: &str, descriptions: &[String]) -> String {
    ELEMENT_SUMMARY_PROMPT
        .replace("{kind}", kind)
        .replace("{name}", name)
        .replace("{descriptions}", &format_descriptions(descriptions))
}

/// Decide whether to call the LLM, concatenate locally, or return as-is.
///
/// Returns `Ok(Some(text))` with the (possibly synthesized) description, or
/// `Ok(None)` when fewer than `min_instances` descriptions exist (caller is
/// expected to keep its existing single description).
pub async fn collapse_descriptions(
    kind: &str,
    name: &str,
    descriptions: &[String],
    config: &ElementSummaryConfig,
    backend: Option<&dyn ChatBackend>,
) -> Result<Option<String>> {
    if !config.enabled {
        return Ok(None);
    }
    if descriptions.len() < config.min_instances {
        return Ok(None);
    }

    let total_chars: usize = descriptions.iter().map(|d| d.len()).sum();
    if total_chars < config.max_chars_for_concat || backend.is_none() {
        // Below cost-guard or no backend available: concatenate locally.
        let joined = descriptions
            .iter()
            .map(|d| d.trim().to_string())
            .filter(|d| !d.is_empty())
            .collect::<Vec<_>>()
            .join(" | ");
        return Ok(Some(joined));
    }

    let prompt = build_summary_prompt(kind, name, descriptions);
    let params = ChatParams {
        max_tokens: Some(config.max_output_tokens),
        temperature: Some(config.temperature),
        num_ctx: None,
    };
    let response = backend.unwrap().complete(&prompt, &params).await?;
    Ok(Some(response.trim().to_string()))
}

/// Collapse a map of `key -> descriptions` into `key -> single description`.
///
/// Keys with fewer than `min_instances` descriptions are dropped from the
/// output (callers should fall back to whatever single description they
/// already have on the element).
pub async fn collapse_all<K: Clone + std::hash::Hash + Eq>(
    kind: &str,
    items: HashMap<K, (String, Vec<String>)>,
    config: &ElementSummaryConfig,
    backend: Option<&dyn ChatBackend>,
) -> Result<HashMap<K, String>> {
    let mut out = HashMap::with_capacity(items.len());
    for (key, (name, descriptions)) in items {
        if let Some(summary) =
            collapse_descriptions(kind, &name, &descriptions, config, backend).await?
        {
            out.insert(key, summary);
        }
    }
    Ok(out)
}

/// Group `EntityData` instances by `(name, type)` and collapse descriptions.
///
/// Returns a map keyed by `(lowercased_name, type)` pointing at the
/// synthesized description for every group meeting the `min_instances`
/// threshold.
pub async fn collapse_entity_descriptions(
    entities: &[crate::entity::prompts::EntityData],
    config: &ElementSummaryConfig,
    backend: Option<&dyn ChatBackend>,
) -> Result<HashMap<(String, String), String>> {
    let mut groups: HashMap<(String, String), (String, Vec<String>)> = HashMap::new();
    for e in entities {
        let key = (e.name.to_lowercase(), e.entity_type.clone());
        let entry = groups
            .entry(key)
            .or_insert_with(|| (e.name.clone(), Vec::new()));
        if !e.description.trim().is_empty() {
            entry.1.push(e.description.clone());
        }
    }
    collapse_all("entity", groups, config, backend).await
}

/// Group `RelationshipData` instances by `(source, target)` and collapse.
///
/// Returns a map keyed by `(lowercased_source, lowercased_target)` pointing
/// at the synthesized description for every group meeting the
/// `min_instances` threshold.
pub async fn collapse_relationship_descriptions(
    relationships: &[crate::entity::prompts::RelationshipData],
    config: &ElementSummaryConfig,
    backend: Option<&dyn ChatBackend>,
) -> Result<HashMap<(String, String), String>> {
    let mut groups: HashMap<(String, String), (String, Vec<String>)> = HashMap::new();
    for r in relationships {
        let key = (r.source.to_lowercase(), r.target.to_lowercase());
        let entry = groups
            .entry(key)
            .or_insert_with(|| (format!("{} -> {}", r.source, r.target), Vec::new()));
        if !r.description.trim().is_empty() {
            entry.1.push(r.description.clone());
        }
    }
    collapse_all("relationship", groups, config, backend).await
}

/// Recording chat backend for tests: stores every prompt seen and returns a
/// canned response.
#[cfg(test)]
pub struct RecordingBackend {
    pub canned: String,
    pub calls: std::sync::Mutex<Vec<String>>,
}

#[cfg(test)]
impl RecordingBackend {
    pub fn new(canned: impl Into<String>) -> Self {
        Self {
            canned: canned.into(),
            calls: std::sync::Mutex::new(Vec::new()),
        }
    }

    pub fn call_count(&self) -> usize {
        self.calls.lock().unwrap().len()
    }
}

#[cfg(test)]
#[async_trait]
impl ChatBackend for RecordingBackend {
    async fn complete(&self, prompt: &str, _params: &ChatParams) -> Result<String> {
        self.calls.lock().unwrap().push(prompt.to_string());
        Ok(self.canned.clone())
    }
}

/// Chat backend that panics if invoked — used to assert the path skipped LLM.
#[cfg(test)]
pub struct PanicBackend;

#[cfg(test)]
#[async_trait]
impl ChatBackend for PanicBackend {
    async fn complete(&self, _prompt: &str, _params: &ChatParams) -> Result<String> {
        panic!("PanicBackend::complete called when LLM should have been skipped");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Single description: returns None so caller keeps the existing one.
    #[tokio::test]
    async fn collapse_single_description_returns_none() {
        let cfg = ElementSummaryConfig::default();
        let backend = RecordingBackend::new("synth");
        let out =
            collapse_descriptions("entity", "Tom", &["only one".into()], &cfg, Some(&backend))
                .await
                .unwrap();
        assert!(out.is_none());
        assert_eq!(backend.call_count(), 0, "must not call LLM for 1 instance");
    }

    /// Below the char budget: concatenate locally without invoking the LLM.
    #[tokio::test]
    async fn collapse_below_budget_concatenates_locally() {
        let cfg = ElementSummaryConfig {
            max_chars_for_concat: 800,
            ..Default::default()
        };
        let backend = RecordingBackend::new("never used");
        let out = collapse_descriptions(
            "entity",
            "Tom",
            &["a young boy".into(), "the protagonist".into()],
            &cfg,
            Some(&backend),
        )
        .await
        .unwrap()
        .unwrap();
        assert!(out.contains("a young boy"));
        assert!(out.contains("the protagonist"));
        assert_eq!(
            backend.call_count(),
            0,
            "below char budget must skip the LLM"
        );
    }

    /// Above the char budget with a backend: call the LLM exactly once.
    #[tokio::test]
    async fn collapse_above_budget_calls_llm() {
        let cfg = ElementSummaryConfig {
            max_chars_for_concat: 50,
            ..Default::default()
        };
        let big = "x".repeat(60);
        let backend = RecordingBackend::new("synthesized");
        let out = collapse_descriptions("entity", "Tom", &[big.clone(), big], &cfg, Some(&backend))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(out, "synthesized");
        assert_eq!(backend.call_count(), 1);
    }

    /// Disabled config: returns None even with multiple instances.
    #[tokio::test]
    async fn collapse_disabled_returns_none() {
        let cfg = ElementSummaryConfig {
            enabled: false,
            ..Default::default()
        };
        let backend = RecordingBackend::new("never used");
        let out = collapse_descriptions(
            "entity",
            "Tom",
            &["a".into(), "b".into()],
            &cfg,
            Some(&backend),
        )
        .await
        .unwrap();
        assert!(out.is_none());
        assert_eq!(backend.call_count(), 0);
    }

    /// `collapse_all` only emits keys that met the `min_instances` threshold.
    #[tokio::test]
    async fn collapse_all_filters_by_min_instances() {
        let cfg = ElementSummaryConfig {
            min_instances: 2,
            ..Default::default()
        };
        let backend = RecordingBackend::new("synth");
        let mut items: HashMap<String, (String, Vec<String>)> = HashMap::new();
        items.insert("tom".into(), ("Tom".into(), vec!["a".into(), "b".into()]));
        items.insert("huck".into(), ("Huck".into(), vec!["only one".into()]));
        let out = collapse_all("entity", items, &cfg, Some(&backend))
            .await
            .unwrap();
        assert!(out.contains_key("tom"));
        assert!(!out.contains_key("huck"));
    }

    /// Below budget without a backend still concatenates locally.
    #[tokio::test]
    async fn collapse_without_backend_below_budget_concatenates() {
        let cfg = ElementSummaryConfig::default();
        let out = collapse_descriptions(
            "entity",
            "Tom",
            &["alpha".into(), "beta".into()],
            &cfg,
            None,
        )
        .await
        .unwrap()
        .unwrap();
        assert!(out.contains("alpha") && out.contains("beta"));
    }

    /// `collapse_entity_descriptions` groups by name+type, drops singletons.
    #[tokio::test]
    async fn collapse_entity_descriptions_groups_by_name_and_type() {
        use crate::entity::prompts::EntityData;
        let cfg = ElementSummaryConfig::default();
        let entities = vec![
            EntityData {
                name: "Tom".into(),
                entity_type: "PERSON".into(),
                description: "a young boy".into(),
            },
            EntityData {
                name: "tom".into(), // case-insensitive grouping
                entity_type: "PERSON".into(),
                description: "the protagonist".into(),
            },
            EntityData {
                name: "Huck".into(),
                entity_type: "PERSON".into(),
                description: "Tom's friend".into(),
            },
        ];
        let backend = RecordingBackend::new("synth");
        let out = collapse_entity_descriptions(&entities, &cfg, Some(&backend))
            .await
            .unwrap();
        assert!(out.contains_key(&("tom".to_string(), "PERSON".to_string())));
        assert!(!out.contains_key(&("huck".to_string(), "PERSON".to_string())));
    }

    /// `collapse_relationship_descriptions` groups by source/target pair.
    #[tokio::test]
    async fn collapse_relationship_descriptions_groups_by_pair() {
        use crate::entity::prompts::RelationshipData;
        let cfg = ElementSummaryConfig::default();
        let relationships = vec![
            RelationshipData {
                source: "Tom".into(),
                target: "Huck".into(),
                description: "best friends".into(),
                strength: 0.9,
            },
            RelationshipData {
                source: "tom".into(),
                target: "huck".into(),
                description: "go on adventures together".into(),
                strength: 0.8,
            },
        ];
        let backend = RecordingBackend::new("synth");
        let out = collapse_relationship_descriptions(&relationships, &cfg, Some(&backend))
            .await
            .unwrap();
        assert!(out.contains_key(&("tom".to_string(), "huck".to_string())));
    }
}
