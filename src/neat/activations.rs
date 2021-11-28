use std::collections::HashMap;
use std::iter::FromIterator;
use super::util::RandRange;
use num_traits::FloatConst;

pub fn sigmoid_f64(z: f64) -> f64 {
    1.0 / (1.0 + f64::exp(-z))
}

pub fn sigmoid_f32(z: f32) -> f32 {
    1.0 / (1.0 + f32::exp(-z))
}

pub fn relu_f64(z: f64) -> f64 {
    z.max(0.0)
}

pub fn relu_f32(z: f32) -> f32 {
    z.max(0.0)
}

pub fn square_f64(z: f64) -> f64 {
    z*z
}

pub fn square_f32(z: f32) -> f32 {
    z*z
}

pub fn inv_f64(z: f64) -> f64 {
    1.0/z
}

pub fn inv_f32(z: f32) -> f32 {
    1.0/z
}

pub fn step_f64(z: f64) -> f64 {
    if z > 0.0 {1.0}else{0.0}
}

pub fn step_f32(z: f32) -> f32 {
    if z > 0.0 {1.0}else{0.0}
}

pub fn neg_f64(z: f64) -> f64 {
    -z
}

pub fn neg_f32(z: f32) -> f32 {
    -z
}

pub fn identity<X>(z: X) -> X {
    z
}

pub fn const_neg1_f32(z: f32) -> f32 {
    -1f32
}
pub fn const_neg1_f64(z: f64) -> f64 {
    -1f64
}

pub fn const1_f32(z: f32) -> f32 {
    1f32
}
pub fn const1_f64(z: f64) -> f64 {
    1f64
}

pub fn const_pi_f32(z: f32) -> f32 {
    f32::PI()
}
pub fn const_pi_f64(z: f64) -> f64 {
    f64::PI()
}

pub fn const_e_f32(z: f32) -> f32 {
    f32::E()
}
pub fn const_e_f64(z: f64) -> f64 {
    f64::E()
}
const sigma32:f32 = 0.5;
const sigma64:f64 = 0.5;

pub fn gaussian32(z: f32) -> f32 {
    1./(f32::PI()*sigma32*sigma32)*f32::exp(-z*z/(2.*sigma32*sigma32))
}
pub fn gaussian64(z: f64) -> f64 {
    1./(f64::PI()*sigma64*sigma64)*f64::exp(-z*z/(2.*sigma64*sigma64))
}



pub const ALL_ACT_FN_NAME: [&'static str; 21] = [
    "identity",
    "inv",
    "sigmoid",
    "relu",
    "sin",
    "cos",
    "tan",
    "tanh",
    "abs",
    "square",
    "step",
    "ln",
    "exp",
    "const1",
    "floor",
    "fract",
    "const1",
    "const_pi",
    "const_e",
    "const_neg1",
    "neg_f32",
];
