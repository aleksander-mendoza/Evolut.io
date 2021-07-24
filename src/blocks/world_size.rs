use crate::blocks::face_orientation::FaceOrientation;

pub const CHUNK_WIDTH: usize = 16;
pub const CHUNK_DEPTH: usize = 16;
pub const CHUNK_HEIGHT: usize = 256;


#[derive(Eq,PartialEq,Clone,Copy)]
pub struct WorldSize {
    width: usize,
    depth: usize,
}

impl WorldSize {
    pub fn new(width: usize, depth: usize)->Self{
        Self{width,depth}
    }

    pub fn absolute_block_to_chunk_block_position(x:usize,y:usize,z:usize) -> (u8,u8,u8) {
        ((x % CHUNK_WIDTH) as u8, y as u8, (z % CHUNK_DEPTH) as u8)
    }
    pub fn chunk_idx_into_chunk_pos(&self, chunk_idx: usize) -> (usize, usize) {
        (chunk_idx % self.width, chunk_idx / self.width)
    }
    pub fn total_chunks(&self) -> usize {
        self.width*self.depth
    }
    pub fn chunk_pos_into_chunk_idx(&self, x: usize, z: usize) -> usize {
        assert!(x < self.width);
        assert!(z < self.depth);
        z * self.width + x
    }
    pub fn block_pos_into_chunk_idx(&self, x: usize, z: usize) -> usize {
        self.chunk_pos_into_chunk_idx(x / CHUNK_WIDTH, z / CHUNK_DEPTH)
    }
    pub fn block_pos_into_world_idx(&self, x: usize, y: usize, z: usize) -> usize {
        assert!(x < self.world_width());
        assert!(z < self.world_depth());
        assert!(y < self.height());
        z * self.world_width() + x + y * self.world_area()
    }
    pub fn world_idx_into_block_pos(&self, idx: usize) -> (usize, usize, usize) {
        ((idx % self.world_area()) / self.world_width(), idx % self.world_width(), idx / self.world_area())
    }
    pub fn world_area(&self) -> usize {
        self.world_width() * self.world_depth()
    }
    pub fn world_volume(&self) -> usize {
        self.world_width() * self.world_depth() * self.height()
    }
    pub fn depth(&self) -> usize {
        self.depth
    }
    pub fn width(&self) -> usize {
        self.width
    }
    pub fn height(&self) -> usize {
        CHUNK_HEIGHT
    }
    pub fn world_depth(&self) -> usize {
        self.depth * CHUNK_DEPTH
    }
    pub fn world_width(&self) -> usize {
        self.width * CHUNK_WIDTH
    }

    pub fn is_position_in_bounds(&self, x: usize, y: usize, z: usize) -> bool {
        y < CHUNK_HEIGHT && x < self.world_width() && z < self.world_depth()
    }
    pub fn is_point_in_bounds(&self, x: f32, y: f32, z: f32) -> bool {
        0f32 <= x && 0f32 <= y && 0f32 <= z && y < CHUNK_HEIGHT as f32 && x < self.world_width() as f32 && z < self.world_depth() as f32
    }
    pub fn for_each_neighbour<F: FnMut(usize, usize, usize, FaceOrientation)>(
        &self,
        x: usize,
        y: usize,
        z: usize,
        mut f: F,
    ) {
        if y + 1 < CHUNK_HEIGHT {
            f(x, y + 1, z, FaceOrientation::YPlus)
        }
        if y >= 1 {
            f(x, y - 1, z, FaceOrientation::YMinus)
        }
        if x + 1< self.world_width() {
            f(x + 1, y, z, FaceOrientation::XPlus)
        }
        if x >= 1 {
            f(x - 1, y, z, FaceOrientation::XMinus)
        }
        if z + 1< self.world_depth() {
            f(x, y, z + 1, FaceOrientation::ZPlus)
        }
        if z >= 1 {
            f(x, y, z - 1, FaceOrientation::ZMinus)
        }
    }
    pub fn point_to_block_position(point: &[f32]) -> (usize, usize, usize) {
        (point[0] as usize, point[1] as usize, point[2] as usize)
    }
}