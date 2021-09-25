use crate::blocks::WorldSize;
use crate::neat::num::Num;
use rand::distributions::Distribution;

pub struct PerlinNoiseMap {
    rand_vals: Vec<f32>,
    world_size: WorldSize,
}

impl PerlinNoiseMap {
    pub fn new_around(world_size: WorldSize, scale:f32, mean: f32, span: f32) -> Self {
        Self::new(world_size,scale,mean - span/2., span)
    }
    pub fn new_between(world_size: WorldSize, scale:f32, min: f32, max: f32) -> Self {
        Self::new(world_size,scale,min, max-min)
    }
    /**scale of 16 would yield one random value per chunk and then each individual block would get some interpolated value. scale of 1 would generate
    one random value per block (so it wouldn't be any perlin noise at all but just a uniform random noise)*/
    pub fn new(world_size: WorldSize, scale:f32, min: f32, size: f32) -> Self {
        let mut rand_vals = Vec::with_capacity(world_size.world_area());
        unsafe {
            rand_vals.set_len(world_size.world_area());
        }
        let size_with_margins = ((world_size.world_width() as f32/scale).ceil() as usize + 2,
                                               (world_size.world_depth() as f32 / scale).ceil() as usize + 2);
        let chunk_vals: Vec<f32> = (0..size_with_margins.0 * size_with_margins.1).map(|_| min + size * f32::random()).collect();
        for x in 0..world_size.world_width() {
            for z in 0..world_size.world_depth() {
                let pos_with_margins_x = x as f32 + /*middle point in the block=0.5*/ 0.5 + /*size of extra chunk on the margin*/scale;
                let pos_with_margins_z = z as f32 + /*middle point in the block=0.5*/ 0.5 + /*size of extra chunk on the margin*/scale;
                let radius = scale / 2.;
                let fraction_x = ((pos_with_margins_x - scale / 2f32) / scale).fract();
                let fraction_z = ((pos_with_margins_z - scale / 2f32) / scale).fract();
                fn idx(world_size:(usize,usize),scale:f32,x:f32,z:f32)->usize{
                    debug_assert!(((x / scale) as usize) < world_size.0, "{},{} / {} < {:?}", z,x,scale, world_size);
                    debug_assert!(((z / scale) as usize) < world_size.1, "{},{} / {}< {:?}", z,x,scale, world_size);
                    (x / scale) as usize * world_size.1 + (z / scale) as usize
                }
                let neighbour_val_right_top = chunk_vals[idx(size_with_margins,scale,pos_with_margins_x + radius, pos_with_margins_z + radius)];
                let neighbour_val_right_bottom = chunk_vals[idx(size_with_margins,scale,pos_with_margins_x + radius, pos_with_margins_z - radius)];
                let neighbour_val_left_top = chunk_vals[idx(size_with_margins,scale,pos_with_margins_x - radius, pos_with_margins_z + radius)];
                let neighbour_val_left_bottom = chunk_vals[idx(size_with_margins,scale,pos_with_margins_x - radius, pos_with_margins_z - radius)];
                let val_left = fraction_z.smoothstep_between(neighbour_val_left_bottom, neighbour_val_left_top);
                let val_right = fraction_z.smoothstep_between(neighbour_val_right_bottom, neighbour_val_right_top);
                let val = fraction_x.smoothstep_between(val_left, val_right);
                rand_vals[world_size.block_pos_xz_into_world_idx(x, z)] = val;
            }
        }

        Self { rand_vals, world_size }
    }
    pub fn val(&self, x: usize, z: usize) -> f32 {
        self.rand_vals[self.world_size.block_pos_xz_into_world_idx(x, z)]
    }
}