use crate::blocks::{FaceOrientation, Block};

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
pub const AIR:Block = Block::new(0);
pub const WATER:Block = Block::new(AIR.id()+1);
pub const LAVA:Block = Block::new(WATER.id()+1);
pub const GLASS:Block = Block::new(LAVA.id()+1);
pub const ICE:Block = Block::new(GLASS.id()+1);
pub const SPAWNER:Block = Block::new(ICE.id()+1);
pub const LEAVES:Block = Block::new(SPAWNER.id()+1);
pub const STONE:Block = Block::new(LEAVES.id()+1);
pub const DIRT:Block = Block::new(STONE.id()+1);
pub const GRASS:Block = Block::new(DIRT.id()+1);
pub const PLANK:Block = Block::new(GRASS.id()+1);
pub const CRAFTING:Block = Block::new(PLANK.id()+1);
pub const SLAB:Block = Block::new(CRAFTING.id()+1);
pub const BRICK:Block = Block::new(SLAB.id()+1);
pub const TNT:Block = Block::new(BRICK.id()+1);
pub const COBBLESTONE:Block = Block::new(TNT.id()+1);
pub const BEDROCK:Block = Block::new(COBBLESTONE.id()+1);
pub const SAND:Block = Block::new(BEDROCK.id()+1);
pub const GRAVEL:Block = Block::new(SAND.id()+1);
pub const WOOD:Block = Block::new(GRAVEL.id()+1);
pub const IRON:Block = Block::new(WOOD.id()+1);
pub const GOLD:Block = Block::new(IRON.id()+1);
pub const DIAMOND:Block = Block::new(GOLD.id()+1);
pub const EMERALD:Block = Block::new(DIAMOND.id()+1);
pub const GOLD_ORE:Block = Block::new(EMERALD.id()+1);
pub const IRON_ORE:Block = Block::new(GOLD_ORE.id()+1);
pub const COAL_ORE:Block = Block::new(IRON_ORE.id()+1);
pub const BOOKSHELF:Block = Block::new(COAL_ORE.id()+1);
pub const MOSS_STONE:Block = Block::new(BOOKSHELF.id()+1);
pub const OBSIDIAN:Block = Block::new(MOSS_STONE.id()+1);
pub const SPONGE:Block = Block::new(OBSIDIAN.id()+1);
pub const DIAMOND_ORE:Block = Block::new(SPONGE.id()+1);
pub const REDSTONE_ORE:Block = Block::new(DIAMOND_ORE.id()+1);
pub const SNOW:Block = Block::new(REDSTONE_ORE.id()+1);
pub const NO_OF_TRAVERSABLE_BLOCKS:u32 = 3;
pub const NO_OF_TRANSPARENT_BLOCKS:u32 = 7;
pub const BLOCKS:[BlockPropExtra;34] = [
    BlockPropExtra::regular_transparent("air", /*Some dummy value*/256, 0., 0.05),
    BlockPropExtra::regular_transparent("water", 31, 0.5, 1.0),
    BlockPropExtra::regular_transparent("lava", 36, 0.5, 3.011),
    // blocks above are traversable. Blocks below are solid (Notice
    // that if a block is traversable, player's camera might get inside it.
    // When that happens the block's faces will be culled. This means that every
    // traversable block must be transparent to prevent a situation where
    // the player could see through walls)
    BlockPropExtra::regular_transparent("glass", 28, 0.09, 0.1),
    BlockPropExtra::regular_transparent("ice", 55, 0.7, 0.9167),
    BlockPropExtra::regular_transparent("spawner", 53, 0.1, 2.710),
    BlockPropExtra::top_sides_bottom_transparent("leaves", 51,52, 51, 0.1, 0.143),
    // blocks above are transparent. Blocks below are not
    BlockPropExtra::regular("stone", 1, 2.26796),
    BlockPropExtra::regular("dirt", 2, 1.3),
    BlockPropExtra::top_sides_bottom("grass", 0, 3,2, 1.4),
    BlockPropExtra::regular("plank", 4, 1.5/4.),
    BlockPropExtra::top_sides_bottom_front("crafting", 59, 62,4, 63, 1.5),
    BlockPropExtra::top_sides_bottom("slab", 6,5,6, 2.26796),
    BlockPropExtra::regular("brick", 7, 1.9),
    BlockPropExtra::top_sides_bottom("tnt", 9,8,10, 1.65),
    BlockPropExtra::regular("cobblestone", 11, 2.26796),
    BlockPropExtra::regular("bedrock", 12, 3.1),
    BlockPropExtra::regular("sand", 13, 1.62),
    BlockPropExtra::regular("gravel", 14, 1.68),
    BlockPropExtra::top_sides_bottom("wood", 16,15,16, 1.5),
    BlockPropExtra::regular("iron", 17, 7.3),
    BlockPropExtra::regular("gold", 18, 19.0),
    BlockPropExtra::regular("diamond", 19, 3.514),
    BlockPropExtra::regular("emerald", 20, 4.),
    BlockPropExtra::regular("gold ore", 21, 2.9),
    BlockPropExtra::regular("iron ore", 22, 2.7),
    BlockPropExtra::regular("coal ore", 23, 2.0),
    BlockPropExtra::regular("bookshelf", 24, 1.5),
    BlockPropExtra::regular("moss stone", 25, 2.26796),
    BlockPropExtra::regular("obsidian", 26, 3.1),
    BlockPropExtra::regular("sponge", 27, 0.1),
    BlockPropExtra::regular("diamond ore", 29, 2.1),
    BlockPropExtra::regular("redstone ore", 30, 2.2),
    BlockPropExtra::regular("snow", 54, 0.05),


];