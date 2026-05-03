//! Natural language processing utilities: semantic chunking, custom NER, and syntax analysis.
pub mod custom_ner;
pub mod semantic_chunking;
pub mod syntax_analyzer;

// Re-export main types
pub use semantic_chunking::{
    ChunkingConfig, ChunkingStats, ChunkingStrategy, SemanticChunk, SemanticChunker,
};

pub use custom_ner::{
    AnnotatedExample, CustomNER, DatasetStatistics, EntityType, ExtractedEntity, ExtractionRule,
    RuleType, TrainingDataset,
};

pub use syntax_analyzer::{
    Dependency, DependencyRelation, NounPhrase, POSTag, SyntaxAnalyzer, SyntaxAnalyzerConfig, Token,
};
