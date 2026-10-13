//! Offline experiment only. Its known real-photo failures prevent adoption.
use clap::Parser;
use image_core::{InpaintMasks, RgbaImage};
use std::path::PathBuf;
mod patch_refine;

#[derive(Parser)]
struct Args {
    #[arg(long)] input: PathBuf,
    #[arg(long)] candidate: PathBuf,
    #[arg(long)] mask: PathBuf,
    #[arg(long)] protection: Option<PathBuf>,
    #[arg(long)] output: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args=Args::parse();
    let source=image::open(args.input)?.into_rgba8();
    let candidate=image::open(args.candidate)?.into_rgba8();
    if source.dimensions()!=candidate.dimensions() { return Err("candidate dimensions differ".into()); }
    let selected=image::open(args.mask)?.into_luma8();
    if selected.dimensions()!=source.dimensions() { return Err("mask dimensions differ".into()); }
    let protection=if let Some(path)=args.protection { let mask=image::open(path)?.into_luma8(); if mask.dimensions()!=source.dimensions(){return Err("protection dimensions differ".into());} mask.into_raw() } else { vec![0;source.width() as usize*source.height() as usize] };
    let source=RgbaImage::new(source.width(),source.height(),source.into_raw())?;
    let candidate=RgbaImage::new(candidate.width(),candidate.height(),candidate.into_raw())?;
    let target=selected.into_raw().into_iter().enumerate().map(|(i,v)|if v>0&&protection[i]==0&&source.pixels[i*4+3]>0{255}else{0}).collect();
    let masks=InpaintMasks{width:source.width,height:source.height,target,protection};
    let refined=patch_refine::refine(&source,&candidate,&masks)?;
    let mut output=source.clone();
    for(i,&v)in masks.target.iter().enumerate(){if v>0{output.pixels[i*4..i*4+3].copy_from_slice(&refined.pixels[i*4..i*4+3]);}}
    image::RgbaImage::from_raw(output.width,output.height,output.pixels).ok_or("invalid output")?.save(args.output)?;
    Ok(())
}
