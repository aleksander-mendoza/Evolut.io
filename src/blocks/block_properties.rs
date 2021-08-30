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
    const fn new(texture_ids:[u32;6], opacity:f32)->Self{
        Self{texture_ids, opacity, mass:0.}
    }
}

impl BlockPropExtra{
    const fn regular_transparent(name:&'static str, texture_id:u32, opacity:f32) ->Self{
        Self{name,prop:BlockProp::new([texture_id;6],opacity)}
    }
    const fn regular(name:&'static str, texture_id:u32)->Self{
        Self::regular_transparent(name,texture_id,1.)
    }
    const fn top_sides_bottom(name:&'static str, texture_id_top:u32,texture_id_side:u32,texture_id_bottom:u32)->Self{
        Self::top_sides_bottom_transparent(name,texture_id_top,texture_id_side,texture_id_bottom,1.)
    }
    const fn top_sides_bottom_transparent(name:&'static str, texture_id_top:u32,texture_id_side:u32,texture_id_bottom:u32,opacity:f32)->Self{
        Self{name,prop:BlockProp::new([texture_id_side,texture_id_side,texture_id_top,texture_id_bottom,texture_id_side,texture_id_side],opacity)}
    }
    const fn top_sides_bottom_front(name:&'static str, texture_id_top:u32,texture_id_side:u32,texture_id_bottom:u32,texture_id_front:u32)->Self{
        Self{name,prop:BlockProp::new([texture_id_side,texture_id_side,texture_id_top,texture_id_bottom,texture_id_side,texture_id_front],1.)}
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
pub const GRAVEL:Block = Block::new(BEDROCK.id()+1);
pub const NO_OF_TRAVERSABLE_BLOCKS:u32 = 3;
pub const NO_OF_TRANSPARENT_BLOCKS:u32 = 7;
pub const BLOCKS:[BlockPropExtra;34] = [
    BlockPropExtra::regular_transparent("air", /*Some dummy value*/256, 0.),
    BlockPropExtra::regular_transparent("water", 31, 0.5),
    BlockPropExtra::regular_transparent("lava", 36, 0.5),
    // blocks above are traversable. Blocks below are solid (Notice
    // that if a block is traversable, player's camera might get inside it.
    // When that happens the block's faces will be culled. This means that every
    // traversable block must be transparent to prevent a situation where
    // the player could see through walls)
    BlockPropExtra::regular_transparent("glass", 28, 0.1),
    BlockPropExtra::regular_transparent("ice", 55, 0.7),
    BlockPropExtra::regular_transparent("spawner", 53, 0.9),
    BlockPropExtra::top_sides_bottom_transparent("leaves", 51,52, 51, 0.8),
    // blocks above are transparent. Blocks below are not
    BlockPropExtra::regular("stone", 1),
    BlockPropExtra::regular("dirt", 2),
    BlockPropExtra::top_sides_bottom("grass", 0, 3,2),
    BlockPropExtra::regular("plank", 4),
    BlockPropExtra::top_sides_bottom_front("crafting", 59, 62,4, 63),
    BlockPropExtra::top_sides_bottom("slab", 6,5,6),
    BlockPropExtra::regular("brick", 7),
    BlockPropExtra::top_sides_bottom("tnt", 9,8,10),
    BlockPropExtra::regular("cobblestone", 11),
    BlockPropExtra::regular("bedrock", 12),
    BlockPropExtra::regular("sand", 13),
    BlockPropExtra::regular("gravel", 14),
    BlockPropExtra::top_sides_bottom("wood", 16,15,16),
    BlockPropExtra::regular("iron", 17),
    BlockPropExtra::regular("gold", 18),
    BlockPropExtra::regular("diamond", 19),
    BlockPropExtra::regular("emerald", 20),
    BlockPropExtra::regular("gold ore", 21),
    BlockPropExtra::regular("iron ore", 22),
    BlockPropExtra::regular("coal ore", 23),
    BlockPropExtra::regular("bookshelf", 24),
    BlockPropExtra::regular("moss stone", 25),
    BlockPropExtra::regular("obsidian", 26),
    BlockPropExtra::regular("sponge", 27),
    BlockPropExtra::regular("diamond ore", 29),
    BlockPropExtra::regular("redstone ore", 30),
    BlockPropExtra::regular("snow", 54),


];