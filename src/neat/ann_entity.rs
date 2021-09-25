pub const ENTITY_MAX_LIDAR_COUNT: usize = 32;
#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C, packed)]
pub struct AnnEntity {
    lidars:[u32;ENTITY_MAX_LIDAR_COUNT],
    lidars_used:u32,
    bone_idx:u32,
    main:u32,
    energy:f32,
    speed:f32,
    max_speed:f32,
}
