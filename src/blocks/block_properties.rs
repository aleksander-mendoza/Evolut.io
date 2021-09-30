use crate::blocks::{FaceOrientation, Block};
use crate::blocks::block::BlockId;

/**This data is visible on CPU*/
pub struct BlockPropExtra{
    pub name:&'static str,
    pub prop:BlockProp
}
/**This data is copied to GPU*/
#[derive(Copy, Clone, Debug)]
#[repr(C, packed)]
pub struct BlockProp{ // this thing needs to be aligned according to GLSL rules!
    texture_ids:[u32;6],
    opacity:f32, // the higher, the more opaque
    mass:f32,
}
impl BlockProp{
    const fn new(texture_ids:[u32;6], opacity:f32, mass:f32)->Self{
        Self{texture_ids, opacity, mass}
    }
}

impl BlockPropExtra{
    const fn regular_transparent(name:&'static str, texture_id:u32, opacity:f32, mass:f32) ->Self{
        Self{name,prop:BlockProp::new([texture_id;6],opacity,mass)}
    }
    const fn regular(name:&'static str, texture_id:u32, mass:f32)->Self{
        Self::regular_transparent(name,texture_id,1.,mass)
    }
    const fn top_sides_bottom(name:&'static str, texture_id_top:u32,texture_id_side:u32,texture_id_bottom:u32, mass:f32)->Self{
        Self::top_sides_bottom_transparent(name,texture_id_top,texture_id_side,texture_id_bottom,1., mass)
    }
    const fn top_sides_bottom_transparent(name:&'static str, texture_id_top:u32,texture_id_side:u32,texture_id_bottom:u32,opacity:f32, mass:f32)->Self{
        Self{name,prop:BlockProp::new([texture_id_side,texture_id_side,texture_id_top,texture_id_bottom,texture_id_side,texture_id_side],opacity, mass)}
    }
    const fn top_sides_bottom_front(name:&'static str, texture_id_top:u32,texture_id_side:u32,texture_id_bottom:u32,texture_id_front:u32, mass:f32)->Self{
        Self{name,prop:BlockProp::new([texture_id_side,texture_id_side,texture_id_top,texture_id_bottom,texture_id_side,texture_id_front],1., mass)}
    }
    pub const fn get_texture_id(&self, ort:FaceOrientation)->u32{
        self.prop.texture_ids[ort as usize]
    }
    pub const fn name(&self)->&'static str{
        self.name
    }
    pub const fn opacity(&self)->f32{
        self.prop.opacity
    }
}
pub const AIR:BlockId = BlockId::new(0);
pub const WATER:BlockId = BlockId::new(1);
pub const LAVA:BlockId = BlockId::new(2);
pub const GLASS:BlockId = BlockId::new(3);
pub const ICE:BlockId = BlockId::new(4);
pub const SWAMP_LEAVES:BlockId = BlockId::new(5);
pub const GOLDEN_LEAVES:BlockId = BlockId::new(6);
pub const OAK_LEAVES:BlockId = BlockId::new(7);
pub const PEACH_LEAVES:BlockId = BlockId::new(8);
pub const AETHER_LEAVES:BlockId = BlockId::new(9);
pub const FROST_LEAVES:BlockId = BlockId::new(10);
pub const STONE:BlockId = BlockId::new(11);
pub const GOLD_ORE:BlockId = BlockId::new(12);
pub const IRON_ORE:BlockId = BlockId::new(13);
pub const COAL_ORE:BlockId = BlockId::new(14);
pub const DIAMOND_ORE:BlockId = BlockId::new(15);
pub const REDSTONE_ORE:BlockId = BlockId::new(16);
pub const DIRT:BlockId = BlockId::new(17);
pub const SWAMP_DIRT:BlockId = BlockId::new(18);
pub const IRON_RICH_DIRT:BlockId = BlockId::new(19);
pub const AETHER_DIRT:BlockId = BlockId::new(20);
pub const FROST_DIRT:BlockId = BlockId::new(21);
pub const GRASS:BlockId = BlockId::new(22);
pub const BERRIES:BlockId = BlockId::new(23);
pub const STRAWBERRIES:BlockId = BlockId::new(24);
pub const WHEAT:BlockId = BlockId::new(25);
pub const SWAMP_GRASS:BlockId = BlockId::new(26);
pub const SWAMP_BERRIES:BlockId = BlockId::new(27);
pub const SWAMP_BACKBERRIES:BlockId = BlockId::new(28);
pub const FROST_GRASS:BlockId = BlockId::new(29);
pub const SNOW_CROCUS:BlockId = BlockId::new(30);
pub const SNOW_BLACKBERRIES:BlockId = BlockId::new(31);
pub const PLANK:BlockId = BlockId::new(32);
pub const SLAB:BlockId = BlockId::new(33);
pub const BRICK:BlockId = BlockId::new(34);
pub const COBBLESTONE:BlockId = BlockId::new(35);
pub const BEDROCK:BlockId = BlockId::new(36);
pub const SAND:BlockId = BlockId::new(37);
pub const RARE_SAND:BlockId = BlockId::new(38);
pub const GRAVEL:BlockId = BlockId::new(39);
pub const WET_GRAVEL:BlockId = BlockId::new(40);
pub const OAK_WOOD:BlockId = BlockId::new(41);
pub const OAK_STEM:BlockId = BlockId::new(42);
pub const PINK_WOOD:BlockId = BlockId::new(43);
pub const PINK_STEM:BlockId = BlockId::new(44);
pub const DARK_WOOD:BlockId = BlockId::new(45);
pub const DARK_STEM:BlockId = BlockId::new(46);
pub const OBSIDIAN:BlockId = BlockId::new(47);
pub const SPONGE:BlockId = BlockId::new(48);
pub const SNOW:BlockId = BlockId::new(49);
pub const NO_OF_TRAVERSABLE_BLOCKS:u32 = 3;
pub const NO_OF_TRANSPARENT_BLOCKS:u32 = 11;

pub const BLOCKS:[BlockPropExtra;50] = [
    BlockPropExtra::regular_transparent("air", /*Some dummy value*/256, 0., 0.05),
    BlockPropExtra::regular_transparent("water", 44, 0.09, 1.0),
    BlockPropExtra::regular_transparent("lava", 45, 0.009, 3.011),
    BlockPropExtra::regular_transparent("glass", 54, 0.1, 0.1),
    BlockPropExtra::regular_transparent("ice", 46, 0.7, 0.9167),
    BlockPropExtra::regular_transparent("swamp_leaves", 47, 0.1, 0.143),
    BlockPropExtra::regular_transparent("golden_leaves", 48, 0.1, 0.143),
    BlockPropExtra::regular_transparent("oak_leaves", 49, 0.1, 0.143),
    BlockPropExtra::regular_transparent("peach_leaves", 50, 0.1, 0.143),
    BlockPropExtra::regular_transparent("aether_leaves", 51, 0.1, 0.143),
    BlockPropExtra::regular_transparent("frost_leaves", 52, 0.1, 0.143),
    BlockPropExtra::regular("stone", 18, 2.26796),
    BlockPropExtra::regular("gold_ore", 19, 2.9),
    BlockPropExtra::regular("iron_ore", 20, 2.7),
    BlockPropExtra::regular("coal_ore", 21, 2.0),
    BlockPropExtra::regular("diamond_ore", 22, 2.1),
    BlockPropExtra::regular("redstone_ore", 23, 2.2),
    BlockPropExtra::regular("dirt", 5, 1.3),
    BlockPropExtra::regular("swamp_dirt", 10, 1.3),
    BlockPropExtra::regular("iron_rich_dirt", 11, 1.3),
    BlockPropExtra::regular("aether_dirt", 12, 1.3),
    BlockPropExtra::regular("frost_dirt", 17, 1.3),
    BlockPropExtra::top_sides_bottom("grass", 3, 4,5, 1.4),
    BlockPropExtra::top_sides_bottom("berries", 0, 4,5, 1.4),
    BlockPropExtra::top_sides_bottom("strawberries", 1, 4,5, 1.4),
    BlockPropExtra::top_sides_bottom("wheat", 2, 4,5, 1.4),
    BlockPropExtra::top_sides_bottom("swamp_grass", 8,9,10, 1.4),
    BlockPropExtra::top_sides_bottom("swamp_berries", 6,9,10, 1.4),
    BlockPropExtra::top_sides_bottom("swamp_backberries", 7,9,10, 1.4),
    BlockPropExtra::top_sides_bottom("frost_grass", 15,16,17, 1.4),
    BlockPropExtra::top_sides_bottom("snow_crocus", 13,16,17, 1.4),
    BlockPropExtra::top_sides_bottom("snow_blackberries", 14,16,17, 1.4),
    BlockPropExtra::regular("plank", 41, 1.5/4.),
    BlockPropExtra::regular("slab", 42, 2.26796),
    BlockPropExtra::regular("brick", 43, 1.9),
    BlockPropExtra::regular("cobblestone", 35, 2.26796),
    BlockPropExtra::regular("bedrock", 36, 3.1),
    BlockPropExtra::regular("sand", 38, 1.62),
    BlockPropExtra::regular("rare_sand", 37, 1.62),
    BlockPropExtra::regular("gravel", 40, 1.68),
    BlockPropExtra::regular("wet_gravel", 39, 1.68),
    BlockPropExtra::top_sides_bottom("oak_wood", 26,25,26, 1.5),
    BlockPropExtra::top_sides_bottom("oak_stem", 26,24,26, 1.5),
    BlockPropExtra::top_sides_bottom("pink_wood", 29,28,29, 1.5),
    BlockPropExtra::top_sides_bottom("pink_stem", 29,27,29, 1.5),
    BlockPropExtra::top_sides_bottom("dark_wood", 32,31,32, 1.5),
    BlockPropExtra::top_sides_bottom("dark_stem", 32,30,32, 1.5),
    BlockPropExtra::regular("obsidian", 33, 3.1),
    BlockPropExtra::regular("sponge", 34, 0.1),
    BlockPropExtra::regular("snow", 15, 0.05)
];