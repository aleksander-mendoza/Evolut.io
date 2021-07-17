pub struct BlockWorld{

}

impl BlockWorld{
    pub fn new() -> Self{
        let frag = include_glsl!("assets/shaders/blocks.frag");
        let vert = include_glsl!("assets/shaders/blocks.vert");
        Self{}
    }
}