use std::ops::{Deref, DerefMut};

#[derive(Copy,Clone, Debug)]
#[repr(align(16))]
pub struct Vec3(pub glm::Vec3);
impl Deref for Vec3{
    type Target = glm::Vec3;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for Vec3{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}