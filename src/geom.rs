//! Geometry for text-line crops: convex hull, minimum-area rectangle,
//! quadrilateral unclip, and a perspective warp back onto the original image.

use image::{Rgb, RgbImage};

#[derive(Clone, Copy, Debug)]
pub struct RotRect {
    pub cx: f32,
    pub cy: f32,
    pub w: f32,
    pub h: f32,
    pub angle: f32,
}

pub fn convex_hull(points: &[(f32, f32)]) -> Vec<(f32, f32)> {
    let mut pts = points.to_vec();
    pts.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    pts.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);
    if pts.len() <= 2 {
        return pts;
    }
    let mut lower = Vec::new();
    for p in &pts {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], *p) <= 0.0 {
            lower.pop();
        }
        lower.push(*p);
    }
    let mut upper = Vec::new();
    for p in pts.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], *p) <= 0.0 {
            upper.pop();
        }
        upper.push(*p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

fn cross(a: (f32, f32), b: (f32, f32), c: (f32, f32)) -> f32 {
    (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)
}

pub fn min_area_rect(points: &[(f32, f32)]) -> Option<RotRect> {
    if points.is_empty() {
        return None;
    }
    let hull = convex_hull(points);
    if hull.len() < 3 {
        let (min_x, max_x, min_y, max_y) = bounds(&hull);
        return Some(RotRect {
            cx: (min_x + max_x) * 0.5,
            cy: (min_y + max_y) * 0.5,
            w: (max_x - min_x).max(1.0),
            h: (max_y - min_y).max(1.0),
            angle: 0.0,
        });
    }
    let mut best: Option<RotRect> = None;
    let mut best_area = f32::MAX;
    let n = hull.len();
    for i in 0..n {
        let (x0, y0) = hull[i];
        let (x1, y1) = hull[(i + 1) % n];
        let dx = x1 - x0;
        let dy = y1 - y0;
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-4 {
            continue;
        }
        let ux = dx / len;
        let uy = dy / len;
        let vx = -uy;
        let vy = ux;
        let mut min_u = f32::MAX;
        let mut max_u = f32::MIN;
        let mut min_v = f32::MAX;
        let mut max_v = f32::MIN;
        for &(x, y) in &hull {
            let u = x * ux + y * uy;
            let v = x * vx + y * vy;
            min_u = min_u.min(u);
            max_u = max_u.max(u);
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }
        let w = (max_u - min_u).max(1.0);
        let h = (max_v - min_v).max(1.0);
        let area = w * h;
        if area < best_area {
            best_area = area;
            let cu = (min_u + max_u) * 0.5;
            let cv = (min_v + max_v) * 0.5;
            best = Some(RotRect {
                cx: cu * ux + cv * vx,
                cy: cu * uy + cv * vy,
                w,
                h,
                angle: uy.atan2(ux),
            });
        }
    }
    best
}

fn bounds(points: &[(f32, f32)]) -> (f32, f32, f32, f32) {
    let mut min_x = f32::MAX;
    let mut max_x = f32::MIN;
    let mut min_y = f32::MAX;
    let mut max_y = f32::MIN;
    for &(x, y) in points {
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }
    if points.is_empty() {
        return (0.0, 1.0, 0.0, 1.0);
    }
    (min_x, max_x, min_y, max_y)
}

/// Expand a rectangle the way DB unclip does: offset by area * ratio / perimeter.
pub fn unclip(rect: RotRect, ratio: f32) -> RotRect {
    let peri = 2.0 * (rect.w + rect.h);
    if peri < 1e-3 {
        return rect;
    }
    let distance = rect.w * rect.h * ratio / peri;
    RotRect {
        w: rect.w + 2.0 * distance,
        h: rect.h + 2.0 * distance,
        ..rect
    }
}

/// Make the long edge the width so recognition sees a horizontal line.
pub fn orient_horizontal(mut rect: RotRect) -> RotRect {
    if rect.h > rect.w {
        std::mem::swap(&mut rect.w, &mut rect.h);
        rect.angle += std::f32::consts::FRAC_PI_2;
    }
    rect
}

pub fn corners(rect: &RotRect) -> [(f32, f32); 4] {
    let (sin, cos) = rect.angle.sin_cos();
    let hw = rect.w * 0.5;
    let hh = rect.h * 0.5;
    let local = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)];
    local.map(|(x, y)| (rect.cx + x * cos - y * sin, rect.cy + x * sin + y * cos))
}

pub fn warp(image: &RgbImage, quad: [(f32, f32); 4], dw: u32, dh: u32) -> Option<RgbImage> {
    if dw < 2 || dh < 2 {
        return None;
    }
    let dest = [
        (0.0, 0.0),
        (dw as f32, 0.0),
        (dw as f32, dh as f32),
        (0.0, dh as f32),
    ];
    let h = homography(&dest, &quad)?;
    let mut out = RgbImage::new(dw, dh);
    let width = image.width() as i32;
    let height = image.height() as i32;
    for y in 0..dh {
        for x in 0..dw {
            let (sx, sy) = apply_h(&h, x as f32 + 0.5, y as f32 + 0.5);
            out.put_pixel(x, y, sample(image, sx, sy, width, height));
        }
    }
    Some(out)
}

fn sample(image: &RgbImage, x: f32, y: f32, width: i32, height: i32) -> Rgb<u8> {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let dx = x - x0 as f32;
    let dy = y - y0 as f32;
    let mut acc = [0.0f32; 3];
    for ky in 0..2 {
        for kx in 0..2 {
            let px = (x0 + kx).clamp(0, width - 1) as u32;
            let py = (y0 + ky).clamp(0, height - 1) as u32;
            let w = if kx == 0 { 1.0 - dx } else { dx } * if ky == 0 { 1.0 - dy } else { dy };
            let p = image.get_pixel(px, py);
            for c in 0..3 {
                acc[c] += p[c] as f32 * w;
            }
        }
    }
    Rgb([
        acc[0].round().clamp(0.0, 255.0) as u8,
        acc[1].round().clamp(0.0, 255.0) as u8,
        acc[2].round().clamp(0.0, 255.0) as u8,
    ])
}

/// Map `from` points onto `to` points. Returns eight coefficients; the last
/// homogeneous element is fixed at 1.
pub fn homography(from: &[(f32, f32); 4], to: &[(f32, f32); 4]) -> Option<[f64; 8]> {
    let mut a = [[0.0f64; 8]; 8];
    let mut b = [0.0f64; 8];
    for i in 0..4 {
        let (u, v) = (from[i].0 as f64, from[i].1 as f64);
        let (x, y) = (to[i].0 as f64, to[i].1 as f64);
        let r0 = i * 2;
        a[r0] = [u, v, 1.0, 0.0, 0.0, 0.0, -u * x, -v * x];
        b[r0] = x;
        a[r0 + 1] = [0.0, 0.0, 0.0, u, v, 1.0, -u * y, -v * y];
        b[r0 + 1] = y;
    }
    solve(&mut a, &mut b)
}

fn solve(a: &mut [[f64; 8]; 8], b: &mut [f64; 8]) -> Option<[f64; 8]> {
    for col in 0..8 {
        let mut pivot = col;
        let mut best = a[col][col].abs();
        for row in (col + 1)..8 {
            let value = a[row][col].abs();
            if value > best {
                best = value;
                pivot = row;
            }
        }
        if best < 1e-10 {
            return None;
        }
        if pivot != col {
            a.swap(pivot, col);
            b.swap(pivot, col);
        }
        let div = a[col][col];
        for k in col..8 {
            a[col][k] /= div;
        }
        b[col] /= div;
        for row in 0..8 {
            if row == col {
                continue;
            }
            let factor = a[row][col];
            if factor == 0.0 {
                continue;
            }
            for k in col..8 {
                a[row][k] -= factor * a[col][k];
            }
            b[row] -= factor * b[col];
        }
    }
    Some(*b)
}

fn apply_h(h: &[f64; 8], u: f32, v: f32) -> (f32, f32) {
    let (u, v) = (u as f64, v as f64);
    let den = h[6] * u + h[7] * v + 1.0;
    let x = (h[0] * u + h[1] * v + h[2]) / den;
    let y = (h[3] * u + h[4] * v + h[5]) / den;
    (x as f32, y as f32)
}

/// Reading order: top to bottom, then left to right, with a small tolerance
/// so a line that slopes stays together.
pub fn sort_reading_order(quads: &mut [([(f32, f32); 4], f32)]) {
    quads.sort_by(|a, b| a.0[0].1.total_cmp(&b.0[0].1).then(a.0[0].0.total_cmp(&b.0[0].0)));
    let n = quads.len();
    for i in 0..n.saturating_sub(1) {
        let mut j = i;
        while j > 0
            && (quads[j].0[0].1 - quads[j - 1].0[0].1).abs() < 10.0
            && quads[j].0[0].0 < quads[j - 1].0[0].0
        {
            quads.swap(j, j - 1);
            j -= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hull_of_a_square_drops_the_center() {
        let points = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0), (1.0, 1.0)];
        let hull = convex_hull(&points);
        assert_eq!(hull.len(), 4);
    }

    #[test]
    fn unclip_grows_both_sides() {
        let rect = RotRect { cx: 0.0, cy: 0.0, w: 40.0, h: 10.0, angle: 0.0 };
        let grown = unclip(rect, 1.5);
        assert!(grown.w > rect.w);
        assert!(grown.h > rect.h);
    }

    #[test]
    fn identity_homography_maps_the_center() {
        let square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        let h = homography(&square, &square).unwrap();
        let (x, y) = apply_h(&h, 5.0, 5.0);
        assert!((x - 5.0).abs() < 1e-3, "{x},{y}");
        assert!((y - 5.0).abs() < 1e-3, "{x},{y}");
    }

    #[test]
    fn warp_copies_a_flat_color() {
        let mut image = RgbImage::new(8, 8);
        for p in image.pixels_mut() {
            *p = Rgb([10, 20, 30]);
        }
        let quad = [(1.0, 1.0), (6.0, 1.0), (6.0, 6.0), (1.0, 6.0)];
        let out = warp(&image, quad, 4, 4).unwrap();
        let p = out.get_pixel(2, 2);
        assert_eq!(p.0, [10, 20, 30]);
    }
}
