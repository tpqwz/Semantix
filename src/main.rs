//! Portable offline gallery: OCR full-text search and CLIP semantic search.

use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand};
use semantix::search::{Hit, Mode};
use semantix::{gui, Config, Engine};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "semantix", version, about = "Offline gallery with OCR and semantic search")]
struct Cli {
    /// Library directory (SQLite, thumbnails, embeddings).
    #[arg(long, global = true)]
    data: Option<PathBuf>,
    /// Directory that holds the ONNX models.
    #[arg(long, global = true)]
    models: Option<PathBuf>,
    /// Run ONNX Runtime on CUDA device 0.
    #[arg(long, global = true)]
    cuda: bool,
    /// Inference device. `cuda` is the same as `--cuda`.
    #[arg(long, global = true, value_parser = ["cpu", "cuda"])]
    device: Option<String>,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Open the gallery window.
    Gui,
    /// Index a folder of images.
    Index { folder: PathBuf },
    /// Search OCR text. Add --semantic or --hybrid for CLIP.
    Search {
        query: Vec<String>,
        #[arg(long)]
        semantic: bool,
        #[arg(long)]
        hybrid: bool,
        /// Run a search stored with save-search. The stored query and mode are used.
        #[arg(long)]
        saved: Option<String>,
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },
    /// Find indexed images similar to a picture.
    Similar {
        image: PathBuf,
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },
    /// Group near-duplicate images by CLIP cosine.
    Duplicates {
        /// Minimum cosine for two pictures to share a group.
        #[arg(long, default_value_t = 0.95)]
        threshold: f32,
        /// Maximum number of groups to print.
        #[arg(long, default_value_t = 50)]
        limit: usize,
    },
    /// Remember a search by name. Later runs use the window or the same query.
    SaveSearch {
        name: String,
        query: Vec<String>,
        #[arg(long)]
        semantic: bool,
        #[arg(long)]
        hybrid: bool,
    },
    /// Print a collection, add indexed files with --add, or drop them with --remove.
    Collection {
        name: String,
        #[arg(long, num_args = 1..)]
        add: Vec<PathBuf>,
        #[arg(long, num_args = 1..)]
        remove: Vec<PathBuf>,
    },
    /// Download the OCR and CLIP models and verify their hashes.
    FetchModels,
}

fn main() {
    semantix::runtime::prepare();
    if let Err(err) = run() {
        eprintln!("semantix: {err:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    let cuda = cli.cuda || cli.device.as_deref() == Some("cuda");
    let cfg = Config::resolve(cli.data, cli.models, cuda);
    match cli.command {
        None | Some(Command::Gui) => gui::run(cfg),
        Some(Command::Index { folder }) => {
            let mut engine = Engine::open(cfg)?;
            for note in engine.notes() {
                eprintln!("{note}");
            }
            engine.index_folder(&folder)?;
            Ok(())
        }
        Some(Command::Search { query, semantic, hybrid, saved, limit }) => {
            let mut engine = Engine::open(cfg)?;
            let (mode, query) = if let Some(name) = saved {
                if !query.is_empty() {
                    return Err(anyhow!("--saved uses the stored query"));
                }
                let found = engine
                    .shelf()?
                    .searches
                    .into_iter()
                    .find(|search| search.name == name)
                    .ok_or_else(|| anyhow!("no saved search named {name}"))?;
                let mode = Mode::parse(&found.mode).ok_or_else(|| anyhow!("saved search {name} has an unknown mode"))?;
                (mode, found.query)
            } else {
                let query = query.join(" ");
                if query.trim().is_empty() {
                    return Err(anyhow!("search needs a query"));
                }
                let mode = if hybrid {
                    Mode::Hybrid
                } else if semantic {
                    Mode::Semantic
                } else {
                    Mode::Text
                };
                (mode, query)
            };
            let hits = match mode {
                Mode::Text => engine.search_text(&query, limit)?,
                Mode::Semantic => engine.search_semantic(&query, limit)?,
                Mode::Hybrid => engine.search_hybrid(&query, limit)?,
            };
            print_hits(&hits);
            Ok(())
        }
        Some(Command::Similar { image, limit }) => {
            let mut engine = Engine::open(cfg)?;
            let hits = engine.similar_path(&image, limit)?;
            print_hits(&hits);
            Ok(())
        }
        Some(Command::Duplicates { threshold, limit }) => {
            let engine = Engine::open(cfg)?;
            let hits = engine.duplicates(threshold, limit)?;
            print_hits(&hits);
            Ok(())
        }
        Some(Command::SaveSearch { name, query, semantic, hybrid }) => {
            let query = query.join(" ");
            if query.trim().is_empty() {
                return Err(anyhow!("save-search needs a query"));
            }
            let mode = if hybrid {
                Mode::Hybrid
            } else if semantic {
                Mode::Semantic
            } else {
                Mode::Text
            };
            let mut engine = Engine::open(cfg)?;
            engine.save_search(&name, mode, &query)?;
            eprintln!("saved {name}");
            Ok(())
        }
        Some(Command::Collection { name, add, remove }) => {
            let mut engine = Engine::open(cfg)?;
            if !remove.is_empty() {
                engine.remove_paths_from_collection(&name, &remove)?;
            }
            let id = if add.is_empty() {
                engine
                    .collection_id(&name)?
                    .ok_or_else(|| anyhow!("no collection named {name}"))?
            } else {
                engine.add_paths_to_collection(&name, &add)?
            };
            let items = engine.collection_items(id)?;
            for item in &items {
                println!("{}", item.path);
            }
            eprintln!("{} files", items.len());
            Ok(())
        }
        Some(Command::FetchModels) => {
            let mut engine = Engine::open(cfg)?;
            for message in engine.fetch_models()? {
                eprintln!("{message}");
            }
            Ok(())
        }
    }
}

fn print_hits(hits: &[Hit]) {
    for hit in hits {
        let snippet = hit.snippet.replace(['\n', '\r', '\t'], " ");
        println!("{:.4}\t{}\t{snippet}", hit.score, hit.path);
    }
    eprintln!("{} matches", hits.len());
}
