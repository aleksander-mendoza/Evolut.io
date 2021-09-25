use crate::pipelines::perlin_noise_map::PerlinNoiseMap;
use crate::blocks::{WorldSize, WorldBlocks, Block};
use crate::blocks::block_properties::*;
use crate::neat::num::Num;

pub const SEA_LEVEL: usize = 128;
pub const FREEZING_TEMPERATURE: f32 = 0.;
pub const SWAMP_HUMIDITY: f32 = 40.;
pub const DESERT_HUMIDITY: f32 = 25.;//humidity of sahara desert
pub const BIOME_PLAINS:usize = 0;
pub const BIOME_DESERT:usize = 1;
pub const BIOME_ARCTIC_DESERT:usize = 2;
pub const BIOME_TAIGA:usize = 3;
pub const BIOME_ARCTIC:usize = 4;
pub const BIOME_SWAMP:usize = 5;
struct BiomeProp{
    dirt:Block,
    grass:Block,
    water_top:Block,
    sea_depth:usize,
    resource_probability: f32,
    resource:fn(f32)->Block,
    surface_artifact_probability: f32,
    surface_artifact:fn(f32)->Block,
}
const BIOMES:[BiomeProp;6] = [
    BiomeProp{dirt:DIRT,grass:GRASS,water_top:WATER,sea_depth:0,resource_probability:0.1,
        resource:|r|
        if r < 0.5{
            BERRIES
        } else if r < 0.75{
            STRAWBERRIES
        } else{
            WHEAT
        },
        surface_artifact_probability: 0.01,
        surface_artifact:|r| if r < 0.8{
            OAK_STEM
        } else{
            PINK_STEM
        }
    },//PLAINS
    BiomeProp{dirt:SAND,grass:SAND,water_top:WATER,sea_depth:0,resource_probability:-1.,
        resource:|r|{unreachable!()},
        surface_artifact_probability: 0.01,
        surface_artifact:|r| {AETHER_LEAVES}
    },//DESERT
    BiomeProp{
        dirt:FROST_DIRT,
        grass:FROST_DIRT,
        water_top:ICE,
        sea_depth:15,
        resource_probability:-1.,
        resource:|r|{unreachable!()},
        surface_artifact_probability: 0.01,
        surface_artifact:|r| {FROST_LEAVES}
    },//ARCTIC_DESERT
    BiomeProp{
        dirt:FROST_DIRT,
        grass:FROST_GRASS,
        water_top:ICE,
        sea_depth:5,
        resource_probability:0.1,
        resource:|r|if r < 0.4{
            SNOW_BLACKBERRIES
        } else {
            SNOW_CROCUS
        },
        surface_artifact_probability: 0.01,
        surface_artifact:|r| if r < 0.8{OAK_STEM}else{FROST_LEAVES}
    },//TAIGA
    BiomeProp{dirt:FROST_DIRT,grass:SNOW,water_top:ICE,sea_depth:15,resource_probability:0.05,
        resource:|r|if r < 0.4{
            SNOW_BLACKBERRIES
        } else {
            SNOW_CROCUS
        },
        surface_artifact_probability: 0.01,
        surface_artifact:|r| {FROST_LEAVES}
    },//ARCTIC
    BiomeProp{dirt:SWAMP_DIRT,grass:SWAMP_GRASS,water_top:WATER,sea_depth:0,resource_probability:0.15,
        resource:|r|if r < 0.5{
            SWAMP_BACKBERRIES
        } else{
            SWAMP_BERRIES
        },
        surface_artifact_probability: 0.01,
        surface_artifact:|r| if r < 0.8{OAK_STEM}else{DARK_STEM}
    },//SWAMP
];

pub struct HeightMap{
    large_scale:PerlinNoiseMap,
    temperature_scale:PerlinNoiseMap,
    humidity_scale:PerlinNoiseMap,
    chunk_scale:PerlinNoiseMap,
    resource_type_scale:PerlinNoiseMap,
    has_resource_scale:PerlinNoiseMap,
}

impl HeightMap{

    pub fn new(world_size:WorldSize)->Self{
        Self{
            large_scale: PerlinNoiseMap::new_around(world_size,64.,128., 64.),
            chunk_scale: PerlinNoiseMap::new_around(world_size,16.,0., 32.),
            temperature_scale: PerlinNoiseMap::new_between(world_size,64.,-30., 70.), // degree celsius
            humidity_scale: PerlinNoiseMap::new_between(world_size,64.,0., 80.), // percentage
            resource_type_scale: PerlinNoiseMap::new_between(world_size,16.,0., 1.),
            has_resource_scale: PerlinNoiseMap::new_between(world_size,4.,0., 1.),
        }
    }
    pub fn biome(&self,x:usize,z:usize)->usize{
        let humidity = self.humidity(x,z);
        let temperature = self.temperature(x,z);
        if temperature < FREEZING_TEMPERATURE{
            if humidity < DESERT_HUMIDITY{
                BIOME_ARCTIC_DESERT
            }else if humidity > SWAMP_HUMIDITY{
                BIOME_ARCTIC
            }else{
                BIOME_TAIGA
            }
        } else {
            if humidity < DESERT_HUMIDITY{
                BIOME_DESERT
            }else if humidity > SWAMP_HUMIDITY{
                BIOME_SWAMP
            }else{
                BIOME_PLAINS
            }
        }
    }
    pub fn height(&self,x:usize,z:usize)->usize{
        (self.chunk_scale.val(x, z)+self.large_scale.val(x, z)) as usize
    }
    pub fn humidity(&self,x:usize,z:usize)->f32{
        self.humidity_scale.val(x, z)
    }
    pub fn temperature(&self,x:usize,z:usize)->f32{
        self.temperature_scale.val(x, z)
    }

    pub fn setup_world_blocks(&self, world_blocks:&mut WorldBlocks) {
        for x in 0..world_blocks.size().world_width() {
            for z in 0..world_blocks.size().world_depth() {
                let height = self.height(x,z);
                let humidity = self.humidity(x,z);
                let temperature = self.temperature(x,z);
                let biome = self.biome(x,z);
                let biome_props = &BIOMES[biome];
                world_blocks.fill_column_to(x, 1, z, height - 4, STONE);
                world_blocks.fill_column_to(x, height - 4, z, height, biome_props.dirt);
                if height < SEA_LEVEL {
                    let sea_depth = SEA_LEVEL - height;
                    world_blocks.fill_column_to(x, height, z, SEA_LEVEL, WATER);
                    world_blocks.set_block(x, SEA_LEVEL, z, if sea_depth < biome_props.sea_depth {biome_props.water_top}else{WATER});
                } else {
                    let has_resource =  self.has_resource_scale.val(x,z) < biome_props.resource_probability;
                    world_blocks.set_block(x, height, z, if has_resource {
                        (biome_props.resource)(self.resource_type_scale.val(x,z))
                    } else {
                        biome_props.grass
                    });
                    if !has_resource{
                        let surface_artifact_prob = f32::random();
                        if surface_artifact_prob < biome_props.surface_artifact_probability{
                            let artifact = (biome_props.surface_artifact)(surface_artifact_prob/biome_props.surface_artifact_probability);
                            world_blocks.set_block(x, height+1, z, artifact);
                        }
                    }
                }
            }
        }
        world_blocks.fill_level(0, 1, BEDROCK);
    }
}