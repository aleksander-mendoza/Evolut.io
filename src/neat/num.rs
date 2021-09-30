use rand::prelude::Distribution;
use std::fmt::Display;
use crate::neat::activations::*;
use crate::neat::util::RandRange;
use core::fmt::Debug;

pub trait Num: Debug + num_traits::Num + Copy + Display + std::ops::AddAssign + std::ops::Sub + std::ops::Div + std::cmp::PartialOrd{
    const ACT_FN_IDENTITY:fn(Self)->Self;
    const ACT_FN_SIGMOID:fn(Self)->Self;
    const ACT_FN_RELU:fn(Self)->Self;
    const ACT_FN_SIN:fn(Self)->Self;
    const ACT_FN_COS:fn(Self)->Self;
    const ACT_FN_TAN:fn(Self)->Self;
    const ACT_FN_TANH:fn(Self)->Self;
    const ACT_FN_ABS:fn(Self)->Self;
    const ACT_FN_SQUARE:fn(Self)->Self;
    const ACT_FN_INV:fn(Self)->Self;
    const ACT_FN_STEP:fn(Self)->Self;
    const ACT_FN_LN:fn(Self)->Self;
    const ACT_FN_EXP:fn(Self)->Self;
    const ACT_FN_GAUSSIAN:fn(Self)->Self;
    const ACT_FN_FLOOR:fn(Self)->Self;
    const ACT_FN_FRACTION:fn(Self)->Self;
    const ACT_FN_CONST_1:fn(Self)->Self;
    const ACT_FN_CONST_PI:fn(Self)->Self;
    const ACT_FN_CONST_E:fn(Self)->Self;
    const ACT_FN_CONST_NEG1:fn(Self)->Self;
    const ACT_FN_NEG:fn(Self)->Self;
    const ALL_ACT_FN: [fn(Self)->Self; 21];
    fn random() -> Self;
    fn random_vec2() -> glm::TVec2<Self>;
    fn random_vec3() -> glm::TVec3<Self>;
    fn random_vec4() -> glm::TVec4<Self>;
    fn random_walk(self) -> Self;

    fn act_fn_name(a_f:fn(Self)->Self)->&'static str{
        Self::ALL_ACT_FN.iter().position(|&f|f==a_f).map(|i|ALL_ACT_FN_NAME[i]).unwrap_or("???")
    }
    fn random_activation_fn() -> fn(Self)->Self;
    fn lerp(self, other:Self, fraction:Self)->Self{
        self + (other - self)*fraction
    }
    fn min(self, other:Self) -> Self{
        if self < other {self} else {other}
    }
    fn max(self, other:Self) -> Self{
        if self > other {self} else {other}
    }
    fn clamp(self, min:Self,max:Self) -> Self{
        self.max(min).min(max)
    }
    fn smoothstep(self) -> Self;
    fn smoothstep_between(self, edge0:Self, edge1:Self) -> Self {
        edge0 + self.smoothstep()*(edge1-edge0)
    }
}
impl Num for f64 {
    const ACT_FN_IDENTITY:fn(f64)->f64 =            identity;
    const ACT_FN_SIGMOID:fn(f64)->f64 =             sigmoid_f64;
    const ACT_FN_RELU:fn(f64)->f64 =                relu_f64;
    const ACT_FN_SIN:fn(f64)->f64 =                 f64::sin;
    const ACT_FN_COS:fn(f64)->f64 =                 f64::cos;
    const ACT_FN_TAN:fn(f64)->f64 =                 f64::tan;
    const ACT_FN_TANH:fn(f64)->f64 =                f64::tanh;
    const ACT_FN_ABS:fn(f64)->f64 =                 f64::abs;
    const ACT_FN_SQUARE:fn(f64)->f64 =              square_f64;
    const ACT_FN_INV:fn(f64)->f64 =                 inv_f64;
    const ACT_FN_STEP:fn(f64)->f64 =                step_f64;
    const ACT_FN_LN:fn(f64)->f64 =                  f64::ln;
    const ACT_FN_EXP:fn(f64)->f64 =                 f64::exp;
    const ACT_FN_GAUSSIAN:fn(f64)->f64 =            const1_f64;
    const ACT_FN_FLOOR:fn(f64)->f64 =               f64::floor;
    const ACT_FN_FRACTION:fn(f64)->f64 =            f64::fract;
    const ACT_FN_CONST_1:fn(f64)->f64 =             const1_f64;
    const ACT_FN_CONST_PI:fn(f64)->f64 =            const_pi_f64;
    const ACT_FN_CONST_E:fn(f64)->f64 =             const_e_f64;
    const ACT_FN_CONST_NEG1:fn(f64)->f64 =          const_neg1_f64;
    const ACT_FN_NEG:fn(f64)->f64 =                 neg_f64;
    const ALL_ACT_FN: [fn(f64)->f64; 21] = [
        identity,
        inv_f64,
        sigmoid_f64,
        relu_f64,
        f64::sin,
        f64::cos,
        f64::tan,
        f64::tanh,
        f64::abs,
        square_f64,
        step_f64,
        f64::ln,
        f64::exp,
        const1_f64,
        f64::floor,
        f64::fract,
        const1_f64,
        const_pi_f64,
        const_e_f64,
        const_neg1_f64,
        neg_f64,
    ];
    fn random() -> Self{
        rand::random()
    }
    fn random_vec2() -> glm::TVec2<Self>{
        glm::vec2(Self::random(),Self::random())
    }
    fn random_vec3() -> glm::TVec3<Self>{
        glm::vec3(Self::random(),Self::random(),Self::random())
    }
    fn random_vec4() -> glm::TVec4<Self>{
        glm::vec4(Self::random(),Self::random(),Self::random(),Self::random())
    }
    fn random_walk(self) -> Self{
        self + Self::random()-0.5
    }
    fn random_activation_fn() -> fn(Self)->Self{
        Self::ALL_ACT_FN[Self::ALL_ACT_FN.len().random()]
    }
    fn smoothstep(self) -> Self {
        self * self * (3. - 2. * self)
    }
}

impl Num for f32 {

    const ACT_FN_IDENTITY:fn(f32)->f32 =            identity;
    const ACT_FN_SIGMOID:fn(f32)->f32 =             sigmoid_f32;
    const ACT_FN_RELU:fn(f32)->f32 =                relu_f32;
    const ACT_FN_SIN:fn(f32)->f32 =                 f32::sin;
    const ACT_FN_COS:fn(f32)->f32 =                 f32::cos;
    const ACT_FN_TAN:fn(f32)->f32 =                 f32::tan;
    const ACT_FN_TANH:fn(f32)->f32 =                f32::tanh;
    const ACT_FN_ABS:fn(f32)->f32 =                 f32::abs;
    const ACT_FN_SQUARE:fn(f32)->f32 =              square_f32;
    const ACT_FN_INV:fn(f32)->f32 =                 inv_f32;
    const ACT_FN_STEP:fn(f32)->f32 =                step_f32;
    const ACT_FN_LN:fn(f32)->f32 =                  f32::ln;
    const ACT_FN_EXP:fn(f32)->f32 =                 f32::exp;
    const ACT_FN_GAUSSIAN:fn(f32)->f32 =            const1_f32;
    const ACT_FN_FLOOR:fn(f32)->f32 =               f32::floor;
    const ACT_FN_FRACTION:fn(f32)->f32 =            f32::fract;
    const ACT_FN_CONST_1:fn(f32)->f32 =             const1_f32;
    const ACT_FN_CONST_PI:fn(f32)->f32 =            const_pi_f32;
    const ACT_FN_CONST_E:fn(f32)->f32 =             const_e_f32;
    const ACT_FN_CONST_NEG1:fn(f32)->f32 =          const_neg1_f32;
    const ACT_FN_NEG:fn(f32)->f32 =                 neg_f32;
    const ALL_ACT_FN: [fn(f32)->f32; 21] = [
        identity,
        inv_f32,
        sigmoid_f32,
        relu_f32,
        f32::sin,
        f32::cos,
        f32::tan,
        f32::tanh,
        f32::abs,
        square_f32,
        step_f32,
        f32::ln,
        f32::exp,
        const1_f32,
        f32::floor,
        f32::fract,
        const1_f32,
        const_pi_f32,
        const_e_f32,
        const_neg1_f32,
        neg_f32,
    ];
    fn random() -> Self{
        rand::random()
    }
    fn random_vec2() -> glm::TVec2<Self>{
        glm::vec2(Self::random(),Self::random())
    }
    fn random_vec3() -> glm::TVec3<Self>{
        glm::vec3(Self::random(),Self::random(),Self::random())
    }
    fn random_vec4() -> glm::TVec4<Self>{
        glm::vec4(Self::random(),Self::random(),Self::random(),Self::random())
    }
    fn random_walk(self) -> Self{
        self + Self::random()-0.5
    }

    fn random_activation_fn() -> fn(Self)->Self{
        Self::ALL_ACT_FN[Self::ALL_ACT_FN.len().random()]
    }
    fn smoothstep(self) -> Self {
        self * self * (3. - 2. * self)
    }
}
