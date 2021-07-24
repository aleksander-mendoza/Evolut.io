use crate::blocks::{FaceOrientation, Block};

pub struct BlockProp{
    name:&'static str,
    texture_ids:[u32;6],
}

impl BlockProp{
    const fn regular(name:&'static str, texture_id:u32)->Self{
        Self{name,texture_ids:[texture_id;6]}
    }
    const fn top_sides_bottom(name:&'static str, texture_id_top:u32,texture_id_side:u32,texture_id_bottom:u32)->Self{
        Self{name,texture_ids:[texture_id_top,texture_id_bottom,texture_id_side,texture_id_side,texture_id_side,texture_id_side]}
    }
    const fn top_sides_bottom_front(name:&'static str, texture_id_top:u32,texture_id_side:u32,texture_id_bottom:u32,texture_id_front:u32)->Self{
        Self{name,texture_ids:[texture_id_top,texture_id_bottom,texture_id_side,texture_id_side,texture_id_side,texture_id_front]}
    }
    pub fn get_texture_id(&self, ort:FaceOrientation)->u32{
        self.texture_ids[ort as usize]
    }
    pub fn name(&self)->&'static str{
        self.name
    }
}
pub const AIR:Block = Block::new(0);
pub const GLASS:Block = Block::new(1);
pub const ICE:Block = Block::new(2);
pub const SPAWNER:Block = Block::new(3);
pub const WATER:Block = Block::new(4);
pub const LEAVES:Block = Block::new(5);
pub const STONE:Block = Block::new(6);
pub const DIRT:Block = Block::new(7);
pub const GRASS:Block = Block::new(8);
pub const PLANK:Block = Block::new(9);
pub const CRAFTING:Block = Block::new(10);
pub const SLAB:Block = Block::new(11);
pub const BRICK:Block = Block::new(12);
pub const TNT:Block = Block::new(13);
pub const COBBLESTONE:Block = Block::new(14);
pub const BEDROCK:Block = Block::new(15);
pub const GRAVEL:Block = Block::new(14);

pub const BLOCKS:[BlockProp;34] = [
    BlockProp::regular("air", /*Some dummy value*/256),
    BlockProp::regular("glass", 28),
    BlockProp::regular("ice", 55),
    BlockProp::regular("spawner", 53),
    BlockProp::regular("water", 31),
    BlockProp::top_sides_bottom("leaves", 51,52, 51),
    // blocks above are transparent. Blocks below are not
    BlockProp::regular("stone", 1),
    BlockProp::regular("dirt", 2),
    BlockProp::top_sides_bottom("grass", 0, 3,2),
    BlockProp::regular("plank", 4),
    BlockProp::top_sides_bottom_front("crafting", 59, 62,4, 63),
    BlockProp::top_sides_bottom("slab", 6,5,6),
    BlockProp::regular("brick", 7),
    BlockProp::top_sides_bottom("tnt", 9,8,10),
    BlockProp::regular("cobblestone", 11),
    BlockProp::regular("bedrock", 12),
    BlockProp::regular("sand", 13),
    BlockProp::regular("gravel", 14),
    BlockProp::top_sides_bottom("wood", 16,15,16),
    BlockProp::regular("iron", 17),
    BlockProp::regular("gold", 18),
    BlockProp::regular("diamond", 19),
    BlockProp::regular("emerald", 20),
    BlockProp::regular("gold ore", 21),
    BlockProp::regular("iron ore", 22),
    BlockProp::regular("coal ore", 23),
    BlockProp::regular("bookshelf", 24),
    BlockProp::regular("moss stone", 25),
    BlockProp::regular("obsidian", 26),
    BlockProp::regular("sponge", 27),
    BlockProp::regular("diamond ore", 29),
    BlockProp::regular("redstone ore", 30),
    BlockProp::regular("lava", 36),
    BlockProp::regular("snow", 54),


];