//! Full-text query shaping and the blend of literal and semantic ranks.

use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Text,
    Semantic,
    Hybrid,
}

impl Mode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Text => "Text",
            Self::Semantic => "Semantic",
            Self::Hybrid => "Hybrid",
        }
    }

    pub fn key(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Semantic => "semantic",
            Self::Hybrid => "hybrid",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "text" => Some(Self::Text),
            "semantic" => Some(Self::Semantic),
            "hybrid" => Some(Self::Hybrid),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Hit {
    pub id: i64,
    pub path: String,
    pub rel: String,
    pub thumb: Option<String>,
    pub snippet: String,
    pub score: f32,
    pub semantic: bool,
    pub width: Option<i64>,
    pub height: Option<i64>,
}

/// Turn a typed query into an FTS5 MATCH string.
///
/// Spaces are AND. A standalone `OR` (any case) stays OR. A leading `-` is
/// NOT. Quotes wrap a phrase. A trailing `*` is a prefix.
pub fn fts_query(input: &str) -> String {
    let mut out = Vec::new();
    let mut rest = input.trim();
    while !rest.is_empty() {
        rest = rest.trim_start();
        if rest.is_empty() {
            break;
        }
        if rest.starts_with('"') {
            let (phrase, next) = match rest[1..].find('"') {
                Some(end) => (&rest[1..end + 1], &rest[end + 2..]),
                None => (&rest[1..], ""),
            };
            if !phrase.is_empty() {
                out.push(quote_token(phrase));
            }
            rest = next;
            continue;
        }
        let end = rest.find(char::is_whitespace).unwrap_or(rest.len());
        let token = &rest[..end];
        rest = &rest[end..];
        if token.eq_ignore_ascii_case("OR") {
            out.push("OR".to_string());
            continue;
        }
        if token.eq_ignore_ascii_case("AND") || token.eq_ignore_ascii_case("NOT") {
            continue;
        }
        if let Some(negated) = token.strip_prefix('-') {
            if !negated.is_empty() {
                out.push(format!("NOT {}", quote_token(negated)));
                continue;
            }
        }
        if let Some(stem) = token.strip_suffix('*') {
            if !stem.is_empty() {
                let cleaned = prefix_stem(stem);
                if !cleaned.is_empty() {
                    out.push(format!("{cleaned}*"));
                }
                continue;
            }
        }
        out.push(quote_token(token));
    }
    while out.first().is_some_and(|part| part == "OR") {
        out.remove(0);
    }
    while out.last().is_some_and(|part| part == "OR") {
        out.pop();
    }
    out.join(" ")
}

fn quote_token(token: &str) -> String {
    format!("\"{}\"", token.replace('"', "\"\""))
}

fn prefix_stem(stem: &str) -> String {
    stem.chars()
        .filter(|ch| !matches!(ch, '"' | '\'' | '(' | ')' | ':' | '^' | '*' | '{' | '}' | '-' | '+'))
        .collect()
}

/// Reciprocal rank fusion, k = 60. A file that ranks well in both lists rises.
pub fn fuse(literal: &[Hit], semantic: &[Hit]) -> Vec<Hit> {
    const K: f32 = 60.0;
    let mut scores: HashMap<i64, (f32, Hit)> = HashMap::new();
    for (rank, hit) in literal.iter().enumerate() {
        let entry = scores.entry(hit.id).or_insert_with(|| (0.0, hit.clone()));
        entry.0 += 1.0 / (K + rank as f32 + 1.0);
    }
    for (rank, hit) in semantic.iter().enumerate() {
        let entry = scores.entry(hit.id).or_insert_with(|| (0.0, hit.clone()));
        entry.0 += 1.0 / (K + rank as f32 + 1.0);
        if entry.1.snippet.is_empty() {
            entry.1.snippet = hit.snippet.clone();
        }
        entry.1.semantic = true;
    }
    let mut out: Vec<Hit> = scores
        .into_values()
        .map(|(score, mut hit)| {
            hit.score = score;
            hit
        })
        .collect();
    out.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.id.cmp(&b.id)));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escapes_fts_tokens() {
        let query = fts_query(r#"alice "chen lee" OR bob -mallory rec*"#);
        assert_eq!(query, r#""alice" "chen lee" OR "bob" NOT "mallory" rec*"#);
    }

    #[test]
    fn fusion_prefers_agreement() {
        let literal = vec![sample(1), sample(2)];
        let semantic = vec![sample(1), sample(3)];
        let fused = fuse(&literal, &semantic);
        assert_eq!(fused[0].id, 1);
        assert!(fused[0].score > fused[1].score);
    }

    fn sample(id: i64) -> Hit {
        Hit {
            id,
            path: format!("p{id}"),
            rel: format!("p{id}"),
            thumb: None,
            snippet: String::new(),
            score: 0.0,
            semantic: false,
            width: None,
            height: None,
        }
    }
}
