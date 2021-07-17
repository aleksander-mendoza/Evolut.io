pub struct BlockWorld{

}

impl BlockWorld{
    pub fn new() -> Self{
        let frag = include_glsl!("assets/shaders/blocks.frag", kind: frag) as &[u32];
        let vert = include_glsl!("assets/shaders/blocks.vert") as &[u32];
        Self{}
    }
}