
#[derive(Debug, Copy, Clone)]
pub struct MvpUniforms {
    pub mvp: glm::Mat4,
    pub mv: glm::Mat4,
}

impl MvpUniforms {
    pub(crate) fn new() -> Self {
        Self {
            mvp: glm::identity(),
            mv: glm::identity(),
        }
    }
}
