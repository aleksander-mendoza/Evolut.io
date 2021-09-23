use crate::pipelines::perlin_noise_map::PerlinNoiseMap;
use crate::blocks::{WorldSize, WorldBlocks};
use crate::blocks::block_properties::*;

pub const SEA_LEVEL: usize = 128;
pub struct HeightMap{
    large_scale:PerlinNoiseMap,
    chunk_scale:PerlinNoiseMap,
    small_scale:PerlinNoiseMap
}

impl HeightMap{

    pub fn new(world_size:WorldSize)->Self{
        Self{
            large_scale: PerlinNoiseMap::new(world_size,64.,128., 64.),
            chunk_scale: PerlinNoiseMap::new(world_size,16.,0., 32.),
            small_scale: PerlinNoiseMap::new(world_size,4.,128., 8.)
        }
    }
    pub fn height(&self,x:usize,z:usize)->usize{
        self.chunk_scale.val(x, z)+self.large_scale.val(x, z)
    }

    pub fn setup_world_blocks(&self, world_blocks:&mut WorldBlocks) {
        for x in 0..world_blocks.size().world_width() {
            for z in 0..world_blocks.size().world_depth() {
                let height = self.height(x,z);
                world_blocks.fill_column_to(x, 1, z, height - 4, STONE);
                world_blocks.fill_column_to(x, height - 4, z, height, DIRT);
                if height < SEA_LEVEL {
                    world_blocks.fill_column_to(x, height, z, SEA_LEVEL + 1, WATER);
                } else {
                    world_blocks.set_block(x, height, z, GRASS);
                }
            }
        }
        world_blocks.fill_level(0, 1, BEDROCK);
    }
}