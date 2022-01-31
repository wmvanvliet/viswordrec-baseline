extern crate lazy_static;
use image::{Rgb, RgbImage, ColorType};
use imageproc::drawing::{draw_text_mut, text_size};
use imageproc::geometric_transformations::{rotate_about_center, Interpolation};
use rusttype::{Font, Scale};
use std::cmp::max;
use rand::prelude::*;
use std::io::prelude::*;
use std::fs::File;
use image::codecs::jpeg::JpegEncoder;
use cpython::{py_fn, py_module_initializer, PyResult, Python};
use std::f32;


py_module_initializer!(render_stimulus, |py, m| {
    m.add(py, "__doc__", "This module is implemented in Rust.")?;
    m.add(py, "render", py_fn!(py, render(text:&str, font_file:&str, font_size:f32, rotation:f32, noise:f64)))?;
    Ok(())
});

const WIDTH:u32 = 224;
const HEIGHT:u32 = 224;

fn render(_: Python, text:&str, font_file:&str, font_size:f32, rotation:f32, noise:f64) -> PyResult<Vec<u8>> {
    let mut image = RgbImage::new(WIDTH, HEIGHT);

    let mut f = File::open(font_file).unwrap();
    let mut buffer = Vec::new();
    f.read_to_end(&mut buffer).unwrap();
    let font = Font::try_from_vec(buffer).unwrap();

    let scale = Scale {
        x: font_size,
        y: font_size,
    };

    let (w, h) = text_size(scale, &font, text);
    draw_text_mut(&mut image, Rgb([120, 120, 120]), (WIDTH as i32 - w) / 2, (HEIGHT as i32 - h) / 2, scale, &font, text);
    let mut image = rotate_about_center(&image, rotation / 180.0 * f32::consts::PI, Interpolation::Bicubic, Rgb([0, 0, 0]));

    let mut rng = rand::thread_rng();

    for pixel in image.pixels_mut() {
        let old_val = pixel.0[0];
        let noise_val = 120.0 + (noise * rng.gen::<f64>() - 0.5) * 220.0;
        let new_val = max(0, 120 - ((1.0 - noise) * old_val as f64) as u8 + noise_val as u8);
        pixel.0 = [new_val, new_val, new_val];
    }

    let mut buffer = Vec::new();
    let mut encoder = JpegEncoder::new(&mut buffer);
    encoder.encode(&image, WIDTH, HEIGHT, ColorType::Rgb8).unwrap();
    Ok(buffer)
}

/*
fn main() {
    render("KOIRA", "/u/45/vanvlm1/unix/projects/reading_models/data/fonts/DejaVuSansMono.ttf", 40.0, 0.0, 0.2);
}
*/
