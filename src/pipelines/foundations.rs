use crate::render::stage_buffer::{StageBuffer, StageSubBuffer, IndirectDispatchSubBuffer, IndirectSubBuffer};
use crate::pipelines::particle::Particle;
use crate::render::command_pool::{CommandPool};


use ash::vk;


use failure::Error;


use crate::render::submitter::{Submitter, fill_submit, fill_zeros_submit};

use crate::render::buffer_type::{Cpu, Storage, GpuIndirect, Uniform};

use crate::blocks::world_size::{CHUNK_VOLUME_IN_CELLS, CHUNK_WIDTH, CHUNK_DEPTH, BROAD_PHASE_CHUNK_VOLUME_IN_CELLS, BROAD_PHASE_CELL_CAPACITY};
use crate::render::subbuffer::SubBuffer;
use crate::pipelines::constraint::Constraint;
use crate::render::buffer::Buffer;
use crate::pipelines::global_mutables::GlobalMutables;
use crate::blocks::{WorldSize, Block, Face, WorldBlocks, WorldFaces};
use crate::render::sampler::Sampler;
use crate::pipelines::bone::Bone;
use crate::blocks::block_properties::{BLOCKS, BlockProp, BEDROCK, DIRT, GRASS, GLASS, PLANK, AIR, STONE, WATER};
use crate::pipelines::sensor::Sensor;
use crate::pipelines::neural_net_layer::NeuralNetLayer;
use crate::pipelines::neural_net_layer::Aggregate::Overwrite;
use crate::pipelines::muscle::Muscle;
use crate::neat::neat::Neat;
use crate::neat::num::Num;
use crate::neat::cppn::CPPN;
use crate::render::device::{QUEUE_IDX_GRAPHICS, QUEUE_IDX_TRANSFER};
use crate::render::host_buffer::HostBuffer;
use crate::pipelines::player_event::PlayerEvent;
use crate::render::compute::{ComputeDescriptorsBuilder, ComputeDescriptors};
use crate::render::specialization_constants::SpecializationConstants;

pub struct Indirect {
    update_particles: SubBuffer<vk::DispatchIndirectCommand, GpuIndirect>,
    per_bone: SubBuffer<vk::DispatchIndirectCommand, GpuIndirect>,
    agent_sensory_input_update: SubBuffer<vk::DispatchIndirectCommand, GpuIndirect>,
    feed_forward_net: SubBuffer<vk::DispatchIndirectCommand, GpuIndirect>,
    draw_bones: SubBuffer<vk::DrawIndirectCommand, GpuIndirect>,
    draw_blocks: SubBuffer<vk::DrawIndirectCommand, GpuIndirect>,
    draw_particles: SubBuffer<vk::DrawIndirectCommand, GpuIndirect>,
    super_indirect_buffer: SubBuffer<u8, GpuIndirect>,
}

impl Indirect {
    fn new(super_indirect_buffer: SubBuffer<u8, GpuIndirect>, indirect_dispatch: &Submitter<IndirectDispatchSubBuffer>, indirect_draw: &Submitter<IndirectSubBuffer>) -> Self {
        let update_particles = indirect_dispatch.gpu().element(0);
        let per_bone = indirect_dispatch.gpu().element(1);
        let agent_sensory_input_update = indirect_dispatch.gpu().element(2);
        let feed_forward_net = indirect_dispatch.gpu().element(3);
        let draw_bones = indirect_draw.gpu().element(0);
        let draw_blocks = indirect_draw.gpu().element(1);
        let draw_particles = indirect_draw.gpu().element(2);
        Self {
            super_indirect_buffer,
            update_particles,
            per_bone,
            agent_sensory_input_update,
            feed_forward_net,
            draw_bones,
            draw_blocks,
            draw_particles,
        }
    }
    pub fn super_buffer(&self) -> &SubBuffer<u8, GpuIndirect> {
        &self.super_indirect_buffer
    }
    pub fn draw_bones(&self) -> &SubBuffer<vk::DrawIndirectCommand, GpuIndirect> {
        &self.draw_bones
    }
    pub fn draw_blocks(&self) -> &SubBuffer<vk::DrawIndirectCommand, GpuIndirect> {
        &self.draw_blocks
    }
    pub fn draw_particles(&self) -> &SubBuffer<vk::DrawIndirectCommand, GpuIndirect> {
        &self.draw_particles
    }
    pub fn update_particles(&self) -> &SubBuffer<vk::DispatchIndirectCommand, GpuIndirect> {
        &self.update_particles
    }
    pub fn broad_phase_collision_detection(&self) -> &SubBuffer<vk::DispatchIndirectCommand, GpuIndirect> {
        &self.per_bone
    }
    pub fn narrow_phase_collision_detection(&self) -> &SubBuffer<vk::DispatchIndirectCommand, GpuIndirect> {
        &self.per_bone
    }
    pub fn broad_phase_collision_detection_cleanup(&self) -> &SubBuffer<vk::DispatchIndirectCommand, GpuIndirect> {
        &self.per_bone
    }
    pub fn feed_forward_net(&self) -> &SubBuffer<vk::DispatchIndirectCommand, GpuIndirect> {
        &self.feed_forward_net
    }
    pub fn agent_sensory_input_update(&self) -> &SubBuffer<vk::DispatchIndirectCommand, GpuIndirect> {
        &self.agent_sensory_input_update
    }
    pub fn update_bones(&self) -> &SubBuffer<vk::DispatchIndirectCommand, GpuIndirect> {
        &self.per_bone
    }
}

pub struct FoundationInitializer {
    specialization_constants: SpecializationConstants,
    world: Submitter<StageSubBuffer<Block, Cpu, Storage>>,
    faces: Submitter<StageSubBuffer<Face, Cpu, Storage>>,
    face_count_per_chunk_buffer: SubBuffer<Face, Storage>,
    opaque_and_transparent_face_buffer: SubBuffer<Face, Storage>,
    world_copy: Submitter<StageSubBuffer<Block, Cpu, Storage>>,
    tmp_faces_copy: SubBuffer<u32, Storage>,
    blocks_to_be_inserted_or_removed: SubBuffer<u32, Storage>,
    faces_to_be_inserted: SubBuffer<Face, Storage>,
    faces_to_be_removed: SubBuffer<u32, Storage>,
    world_blocks_to_update: SubBuffer<u32, Storage>,
    world_blocks_to_update_copy: SubBuffer<u32, Storage>,
    player_event_uniform: HostBuffer<PlayerEvent, Uniform>,
    collision_grid: SubBuffer<u32, Storage>,
    bones: Submitter<StageSubBuffer<Bone, Cpu, Storage>>,
    persistent_floats: SubBuffer<f32, Storage>,
    neural_net_layers: Submitter<StageSubBuffer<NeuralNetLayer, Cpu, Storage>>,
    global_mutables: Submitter<StageSubBuffer<GlobalMutables, Cpu, Storage>>,
    indirect_dispatch: Submitter<IndirectDispatchSubBuffer>,
    indirect_draw: Submitter<IndirectSubBuffer>,
    indirect: Indirect,
    sampler: Sampler,
    world_size: WorldSize,
}

struct FoundationsCapacity {
    world_size: WorldSize,
    faces_to_be_inserted_chunk_capacity: u32,
    faces_to_be_removed_chunk_capacity: u32,
    max_bones: u64,
    max_faces: u64,
    max_faces_copy: u64,
    max_persistent_floats: u64,
    max_neural_net_layers: u64,
    max_tmp_faces_copy: u64,
    max_world_blocks_to_update: u64,
    max_blocks_to_be_inserted_or_removed: u64,
    max_faces_to_be_inserted: u64,
    max_faces_to_be_removed: u64,
    max_sensors: u64,
}

impl FoundationsCapacity {
    pub fn new(x: usize, z: usize) -> Self {
        let world_size = WorldSize::new(x, z);
        let faces_to_be_inserted_chunk_capacity = 128;
        let faces_to_be_removed_chunk_capacity = 128;
        Self {
            faces_to_be_inserted_chunk_capacity,
            faces_to_be_removed_chunk_capacity,
            max_bones: 1024u64,
            max_faces: 16 * 1024u64 * world_size.total_chunks() as u64,
            max_persistent_floats: 64 * 1024u64, // used as backing memory for vectors, matrices and
            // tensors that make up various neural networks. Especially, the outputs of recursive neural networks
            // often need to be persistent, because those outputs are later fed as inputs to the same neural net.
            max_neural_net_layers: 128u64,
            max_tmp_faces_copy: world_size.world_volume() as u64 / 4,
            max_world_blocks_to_update: world_size.world_volume() as u64 / 4,
            max_blocks_to_be_inserted_or_removed: world_size.world_volume() as u64 / 16,
            max_faces_to_be_inserted: faces_to_be_inserted_chunk_capacity as u64 * world_size.total_chunks() as u64,
            max_faces_to_be_removed: faces_to_be_removed_chunk_capacity as u64 * world_size.total_chunks() as u64,
            max_sensors: 0u64,
            max_faces_copy: 1024u64 * world_size.total_chunks() as u64,
            world_size,
        }
    }
    fn grid_size(&self) -> u64 {
        (self.world_size.total_chunks() * BROAD_PHASE_CHUNK_VOLUME_IN_CELLS) as u64
    }
}

const SUBSTRATE_IN_DIM: usize = 2;
const SUBSTRATE_OUT_DIM: usize = 2;
const SUBSTRATE_WEIGHT_DIM: usize = 1;

struct ZombieNeat<X: Num> {
    substrate_in_positions: Vec<[X; SUBSTRATE_IN_DIM]>,
    substrate_out_positions: Vec<[X; SUBSTRATE_OUT_DIM]>,
    neat: Neat<X>,
    population: Vec<ZombieBrain<X>>,
}

impl<X: Num> ZombieNeat<X> {
    fn into_brain(zombie: Zombie, cppn: CPPN<X>,
                  cmd_pool: &CommandPool,
                  substrate_in_positions: &Vec<[X; SUBSTRATE_IN_DIM]>,
                  substrate_out_positions: &Vec<[X; SUBSTRATE_OUT_DIM]>,
                  persistent_floats_buffer: &SubBuffer<X, Storage>) -> Result<ZombieBrain<X>, vk::Result> {
        let net = cppn.build_feed_forward_net();
        let mut weights = Vec::with_capacity(zombie.weights_len as usize);
        for i in 0..(zombie.sensors_len + zombie.recurrent_len) as usize {
            for o in 0..(zombie.muscles_constraints_len + zombie.recurrent_len) as usize {
                let mut weight = [X::zero()];
                net.run(&[substrate_in_positions[i][0], substrate_in_positions[i][1],
                    substrate_out_positions[o][0], substrate_out_positions[o][1]], &mut weight);
                weights.push(weight[0]);
            }
        }

        let buff = StageBuffer::wrap(cmd_pool, weights.as_slice(), persistent_floats_buffer.sub_elem(zombie.weights_offset as u64, zombie.weights_len as u64))?;
        Ok(ZombieBrain { zombie, cppn, buff })
    }
    fn new(zombies: Vec<Zombie>, cmd_pool: &CommandPool, persistent_floats: &SubBuffer<X, Storage>) -> Result<Self, vk::Result> {
        let substrate_in_positions = (0..zombies[0].sensors_len + zombies[0].recurrent_len).map(|_| {
            let mut arr = [X::zero(); SUBSTRATE_IN_DIM];
            for a in &mut arr { *a = X::random() }
            arr
        }).collect::<Vec<[X; SUBSTRATE_IN_DIM]>>();
        let substrate_out_positions = (0..zombies[0].muscles_constraints_len + zombies[0].recurrent_len).map(|_| {
            let mut arr = [X::zero(); SUBSTRATE_IN_DIM];
            for a in &mut arr { *a = X::random() }
            arr
        }).collect::<Vec<[X; SUBSTRATE_OUT_DIM]>>();
        let mut neat = Neat::new(vec![X::ACT_FN_ABS, X::ACT_FN_SIN, X::ACT_FN_CONST_1, X::ACT_FN_TANH, X::ACT_FN_GAUSSIAN, X::ACT_FN_SQUARE], SUBSTRATE_IN_DIM + SUBSTRATE_OUT_DIM, SUBSTRATE_WEIGHT_DIM);
        let mut cppns = neat.new_cppns(zombies.len());
        for _ in 0..32 {
            for cppn in &mut cppns {
                neat.mutate(cppn, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1);
            }
        }
        // println!("{}",cppns[0]);
        let zombies: Result<Vec<ZombieBrain<X>>, vk::Result> = zombies.into_iter().zip(cppns.into_iter()).map(|(z, c)| Self::into_brain(z, c,
                                                                                                                                        cmd_pool,
                                                                                                                                        &substrate_in_positions,
                                                                                                                                        &substrate_out_positions,
                                                                                                                                        persistent_floats,
        )).collect();
        let population = zombies?;
        // println!("{:?}",(&population[0]).buff.as_slice());
        Ok(Self {
            substrate_in_positions,
            substrate_out_positions,
            neat,
            population,
        })
    }
}

struct Zombie {
    solid_offset: u32,
    phantom_offset: u32,
    weights_offset: u32,
    weights_len: u32,
    sensors_offset: u32,
    sensors_len: u32,
    recurrent_offset: u32,
    recurrent_len: u32,
    muscles_constraints_offset: u32,
    muscles_constraints_len: u32,
}

struct ZombieBrain<X: Num> {
    zombie: Zombie,
    cppn: CPPN<X>,
    buff: Submitter<StageSubBuffer<X, Cpu, Storage>>,
}

struct FoundationsData {
    world_blocks: WorldBlocks,
    world_faces: WorldFaces,
    constraints_data: Vec<Constraint>,
    bone_data: Vec<Bone>,
    sensor_data: Vec<Sensor>,
    muscle_data: Vec<Muscle>,
    block_properties_data: Vec<BlockProp>,
    neural_net_layer_data: Vec<NeuralNetLayer>,
    persistent_floats_len: u32,
}

struct HeightMap {
    heights: Vec<usize>,
    world_size: WorldSize,
}

impl HeightMap {
    pub fn new(world_size: WorldSize, mean: f32, variance: f32) -> Self {
        let mut heights = Vec::with_capacity(world_size.world_area());
        unsafe {
            heights.set_len(world_size.world_area());
        }
        let size_with_margins = WorldSize::new(world_size.width() + 2, world_size.depth() + 2);
        let chunk_heights: Vec<f32> = (0..size_with_margins.total_chunks()).map(|_| mean + f32::random() * variance).collect();
        for x in 0..world_size.world_width() {
            for z in 0..world_size.world_depth() {
                let pos_with_margins_x = x + CHUNK_WIDTH;
                let pos_with_margins_z = z + CHUNK_DEPTH;
                let radius_x = CHUNK_WIDTH / 2 - 1;
                let radius_z = CHUNK_DEPTH / 2 - 1;
                let fraction_x = ((pos_with_margins_x as f32 + 0.5 - CHUNK_WIDTH as f32 / 2f32) / (CHUNK_WIDTH as f32)).fract();
                let fraction_z = ((pos_with_margins_z as f32 + 0.5 - CHUNK_DEPTH as f32 / 2f32) / (CHUNK_DEPTH as f32)).fract();
                let neighbour_height_right_top = chunk_heights[size_with_margins.block_pos_into_chunk_idx(pos_with_margins_x + radius_x, pos_with_margins_z + radius_z)];
                let neighbour_height_right_bottom = chunk_heights[size_with_margins.block_pos_into_chunk_idx(pos_with_margins_x + radius_x, pos_with_margins_z - radius_z)];
                let neighbour_height_left_top = chunk_heights[size_with_margins.block_pos_into_chunk_idx(pos_with_margins_x - radius_x, pos_with_margins_z + radius_z)];
                let neighbour_height_left_bottom = chunk_heights[size_with_margins.block_pos_into_chunk_idx(pos_with_margins_x - radius_x, pos_with_margins_z - radius_z)];
                let height_left = fraction_z.smoothstep_between(neighbour_height_left_bottom, neighbour_height_left_top);
                let height_right = fraction_z.smoothstep_between(neighbour_height_right_bottom, neighbour_height_right_top);
                let height = fraction_x.smoothstep_between(height_left, height_right) as usize;
                heights[world_size.block_pos_xz_into_world_idx(x, z)] = height;
            }
        }
        Self { heights, world_size }
    }
    pub fn height(&self, x: usize, z: usize) -> usize {
        self.heights[self.world_size.block_pos_xz_into_world_idx(x, z)]
    }
}

impl FoundationsData {
    pub fn new(cap: &FoundationsCapacity) -> Self {
        Self {
            world_blocks: WorldBlocks::new(cap.world_size.clone()),
            world_faces: WorldFaces::with_capacity(cap.world_size.clone(), cap.max_faces as usize),
            // particles_data: std::iter::repeat_with(Particle::random).take((cap.solid_particles+cap.phantom_particles) as usize).collect(),
            constraints_data: vec![],
            bone_data: vec![],
            sensor_data: vec![],
            muscle_data: vec![],
            neural_net_layer_data: vec![],
            block_properties_data: BLOCKS.iter().map(|p| p.prop).collect(),
            persistent_floats_len: 0,
        }
    }


    fn setup_world_blocks(&mut self, heights: &HeightMap, sea_level: usize) {
        let size_with_margins = WorldSize::new(self.world_blocks.size().width() + 2, self.world_blocks.size().depth() + 2);
        for x in 0..self.world_blocks.size().world_width() {
            for z in 0..self.world_blocks.size().world_depth() {
                let height = heights.height(x, z);
                self.world_blocks.fill_column_to(x, 1, z, height - 4, STONE);
                self.world_blocks.fill_column_to(x, height - 4, z, height, DIRT);
                if height < sea_level {
                    self.world_blocks.fill_column_to(x, height, z, sea_level + 1, WATER);
                } else {
                    self.world_blocks.set_block(x, height, z, GRASS);
                }
            }
        }
        self.world_blocks.fill_level(0, 1, BEDROCK);
    }
    fn setup_world_faces(&mut self) {
        let Self { world_blocks, world_faces, .. } = self;
        world_faces.generate_faces(world_blocks);
    }

    fn compute_constants(&self, cap: &FoundationsCapacity) -> GlobalMutables {
        let FoundationsCapacity { world_size, .. } = cap;
        let Self { sensor_data, bone_data, .. } = self;
        GlobalMutables {
            blocks_to_be_inserted_or_removed: 0,
            bones: bone_data.len() as u32,
            dummy1: 0,
            dummy2: 0,
            held_bone_idx: 0,
            world_blocks_to_update: 0,
            ambience_tick: 0,
            new_world_blocks_to_update: 0,
        }
    }
}
struct CollisionCell{
    len:u32,
    contents:[u32;BROAD_PHASE_CELL_CAPACITY]
}
fn append_owned<X>(v: &mut Vec<X>, mut v2: Vec<X>) {
    v.append(&mut v2);
}

impl FoundationInitializer {
    pub fn faces(&self) -> &StageSubBuffer<Face, Cpu, Storage> {
        &self.faces
    }

    pub fn face_count_per_chunk_buffer(&self) -> &SubBuffer<Face, Storage> {
        &self.face_count_per_chunk_buffer
    }
    pub fn opaque_and_transparent_face_buffer(&self) -> &SubBuffer<Face, Storage> {
        &self.opaque_and_transparent_face_buffer
    }
    pub fn tmp_faces_copy(&self) -> &SubBuffer<u32, Storage> {
        &self.tmp_faces_copy
    }
    pub fn global_mutables(&self) -> &StageSubBuffer<GlobalMutables, Cpu, Storage> {
        &self.global_mutables
    }
    pub fn specialization_constants(&self) -> &SpecializationConstants{
        &self.specialization_constants
    }
    pub fn persistent_floats(&self) -> &SubBuffer<f32, Storage> {
        &self.persistent_floats
    }
    pub fn collision_grid(&self) -> &SubBuffer<u32, Storage> {
        &self.collision_grid
    }
    pub fn bones(&self) -> &StageSubBuffer<Bone, Cpu, Storage> {
        &self.bones
    }

    pub fn indirect(&self) -> &Indirect {
        &self.indirect
    }
    pub fn sampler(&self) -> &Sampler {
        &self.sampler
    }
    pub fn world_size(&self) -> &WorldSize {
        &self.world_size
    }
    pub fn world(&self) -> &StageSubBuffer<Block, Cpu, Storage> {
        &self.world
    }

    pub fn new(cmd_pool: &CommandPool) -> Result<Self, failure::Error> {
        let cap = FoundationsCapacity::new(2, 2);
        let mut data = FoundationsData::new(&cap);
        let heights = HeightMap::new(cap.world_size, 100., 32.);
        data.setup_world_blocks(&heights, 112);
        data.setup_world_faces();
        // data.particle_data.extend((0..64).map(|_| Particle::random()));
        for _ in 0..8 {
            let x = rand::random::<usize>() % cap.world_size.world_width();
            let z = rand::random::<usize>() % cap.world_size.world_depth();
            data.bone_data.push(Bone::new(glm::vec3(x as f32, heights.height(x, z) as f32 + 5., z as f32), 0.5, glm::vec3(1., 0., 0.), 1., 1.0));
        }

        data.neural_net_layer_data.append(&mut vec![
            NeuralNetLayer::new_input_recurrent(0, 0, 0, 0),
            NeuralNetLayer::new_hidden(0, 0, 0, 0, 0, None, Overwrite),
            NeuralNetLayer::new_output(0, 0, 0, 0, 0, 0)
        ]);
        // let mut zombies = vec![];
        // for r in 0..5 {
        //     for c in 0..1 {
        //         zombies.push(data.add_zombie(glm::vec3(2. + r as f32 * 3., 5., 2. + c as f32 * 3.), &cap));
        //     }
        // }


        let mutables = data.compute_constants(&cap);

        let bones_in_bytes = std::mem::size_of::<Bone>() as u64 * cap.max_bones;
        let faces_in_bytes = std::mem::size_of::<Face>() as u64 * cap.max_faces;
        let tmp_faces_copy_in_bytes = std::mem::size_of::<u32>() as u64 * 3 * cap.max_faces_copy;
        let grid_in_bytes = std::mem::size_of::<CollisionCell>() as u64 * cap.grid_size();
        let world_in_bytes = (std::mem::size_of::<Block>() * cap.world_size.world_volume()) as u64;
        let world_copy_in_bytes = world_in_bytes;
        let world_blocks_to_update_in_bytes = std::mem::size_of::<u32>() as u64 * cap.max_world_blocks_to_update;
        let world_blocks_to_update_copy_in_bytes = world_blocks_to_update_in_bytes;
        let blocks_to_be_inserted_or_removed_in_bytes = std::mem::size_of::<u32>() as u64 * cap.max_blocks_to_be_inserted_or_removed;
        let global_mutables_in_bytes = std::mem::size_of_val(&mutables) as u64;
        let persistent_floats_in_bytes = std::mem::size_of::<f32>() as u64 * cap.max_persistent_floats;
        let neural_net_layers_in_bytes = std::mem::size_of::<NeuralNetLayer>() as u64 * cap.max_neural_net_layers;
        let faces_to_be_inserted_in_bytes = std::mem::size_of::<Face>() as u64 * cap.max_faces_to_be_inserted;
        let faces_to_be_removed_in_bytes = std::mem::size_of::<u32>() as u64 * cap.max_faces_to_be_removed;

        let super_buffer: SubBuffer<u8, Storage> = SubBuffer::with_capacity(cmd_pool.device(),
                                                                            bones_in_bytes +
                                                                                faces_in_bytes +
                                                                                tmp_faces_copy_in_bytes +
                                                                                grid_in_bytes +
                                                                                world_in_bytes +
                                                                                world_copy_in_bytes +
                                                                                world_blocks_to_update_in_bytes +
                                                                                world_blocks_to_update_copy_in_bytes +
                                                                                blocks_to_be_inserted_or_removed_in_bytes +
                                                                                global_mutables_in_bytes +
                                                                                persistent_floats_in_bytes +
                                                                                neural_net_layers_in_bytes +
                                                                                faces_to_be_inserted_in_bytes +
                                                                                faces_to_be_removed_in_bytes,
        )?;
        let offset = 0;
        let bones_buffer = super_buffer.sub(offset..offset + bones_in_bytes).reinterpret_into::<Bone>();
        let offset = offset + bones_in_bytes;
        assert_eq!(offset % 16, 0);
        let face_buffer = super_buffer.sub(offset..offset + faces_in_bytes).reinterpret_into::<Face>();
        let offset = offset + faces_in_bytes ;
        assert_eq!(offset % 16, 0);
        let tmp_faces_copy_buffer = super_buffer.sub(offset..offset + tmp_faces_copy_in_bytes).reinterpret_into::<u32>();
        let offset = offset + tmp_faces_copy_in_bytes;
        assert_eq!(offset % 16, 0);
        let grid_buffer = super_buffer.sub(offset..offset + grid_in_bytes).reinterpret_into::<u32>();
        let offset = offset + grid_in_bytes;
        assert_eq!(offset % 16, 0);
        let world_buffer = super_buffer.sub(offset..offset + world_in_bytes).reinterpret_into::<Block>();
        let offset = offset + world_in_bytes;
        assert_eq!(offset % 16, 0);
        let world_copy_buffer = super_buffer.sub(offset..offset + world_copy_in_bytes).reinterpret_into::<Block>();
        let offset = offset + world_copy_in_bytes;
        assert_eq!(offset % 16, 0);
        let world_blocks_to_update_buffer = super_buffer.sub(offset..offset + world_blocks_to_update_in_bytes).reinterpret_into::<u32>();
        let offset = offset + world_blocks_to_update_in_bytes;
        assert_eq!(offset % 16, 0);
        let world_blocks_to_update_copy_buffer = super_buffer.sub(offset..offset + world_blocks_to_update_copy_in_bytes).reinterpret_into::<u32>();
        let offset = offset + world_blocks_to_update_copy_in_bytes;
        assert_eq!(offset % 16, 0);
        let blocks_to_be_inserted_or_removed_buffer = super_buffer.sub(offset..offset + blocks_to_be_inserted_or_removed_in_bytes).reinterpret_into::<u32>();
        let offset = offset + blocks_to_be_inserted_or_removed_in_bytes;
        assert_eq!(offset % 16, 0);
        let global_mutables_buffer = super_buffer.sub(offset..offset + global_mutables_in_bytes).reinterpret_into::<GlobalMutables>();
        let offset = offset + global_mutables_in_bytes;
        assert_eq!(offset % 16, 0);
        let persistent_floats_buffer = super_buffer.sub(offset..offset + persistent_floats_in_bytes).reinterpret_into::<f32>();
        let offset = offset + persistent_floats_in_bytes;
        assert_eq!(offset % 16, 0);
        let neural_net_layers_buffer = super_buffer.sub(offset..offset + neural_net_layers_in_bytes).reinterpret_into::<NeuralNetLayer>();
        let offset = offset + neural_net_layers_in_bytes;
        assert_eq!(offset % 16, 0);
        let faces_to_be_inserted_buffer = super_buffer.sub(offset..offset + faces_to_be_inserted_in_bytes).reinterpret_into::<Face>();
        let offset = offset + faces_to_be_inserted_in_bytes;
        assert_eq!(offset % 16, 0);
        let faces_to_be_removed_buffer = super_buffer.sub(offset..offset + faces_to_be_removed_in_bytes).reinterpret_into::<u32>();
        let offset = offset + faces_to_be_removed_in_bytes;
        assert_eq!(offset % 16, 0);

        println!("{:?}",data.bone_data[0]);

        let global_mutables = StageBuffer::wrap(cmd_pool, &[mutables], global_mutables_buffer)?;

        let face_count_per_chunk_buffer = face_buffer.sub(..std::mem::size_of::<Face>() as u64 * cap.world_size.total_chunks() as u64 * 2);
        let opaque_and_transparent_face_buffer = face_buffer.sub(std::mem::size_of::<Face>() as u64 * cap.world_size.total_chunks() as u64 * 2..);
        let faces = StageBuffer::wrap(cmd_pool, data.world_faces.as_slice(), face_buffer)?;

        let bones = StageBuffer::wrap(cmd_pool, &data.bone_data, bones_buffer)?;

        let persistent_floats = persistent_floats_buffer;

        let world = StageBuffer::wrap(cmd_pool, data.world_blocks.as_slice(), world_buffer)?;

        let world_copy = StageBuffer::wrap(cmd_pool, data.world_blocks.as_slice(), world_copy_buffer)?;

        let neural_net_layers = StageBuffer::wrap(cmd_pool, &data.neural_net_layer_data, neural_net_layers_buffer)?;

        let sampler = Sampler::new(cmd_pool.device(), vk::Filter::NEAREST, true)?;

        fn dispatch_indirect(x: usize) -> vk::DispatchIndirectCommand {
            vk::DispatchIndirectCommand {
                x: (x as f32 / 32.).ceil() as u32,
                y: 1,
                z: 1,
            }
        }
        fn draw_indirect(vertex_count: u32, instance_count: u32) -> vk::DrawIndirectCommand {
            vk::DrawIndirectCommand {
                vertex_count,
                instance_count,
                first_vertex: 0,
                first_instance: 0,
            }
        }
        let indirect_dispatch_data = vec![
            dispatch_indirect(1),// update_particles.comp
            dispatch_indirect(data.bone_data.len()),// broad_phase_collision_detection.comp broad_phase_collision_detection_cleanup.comp narrow_phase_collision_detection.comp update_bones.comp
            dispatch_indirect(data.sensor_data.len()), // agent_sensory_input_update.comp
            vk::DispatchIndirectCommand {  // feed_forward_net.comp
                x: (data.neural_net_layer_data.len() / 3) as u32,
                y: 1,
                z: 1,
            }
        ];
        let indirect_draw_data = vec![
            draw_indirect(36, data.bone_data.len() as u32),// bones.vert
            draw_indirect(6, data.world_faces.len() as u32),// block.vert
            draw_indirect(0, 1),// particles.vert
        ];
        let indirect_dispatch_in_bytes = std::mem::size_of_val(indirect_dispatch_data.as_slice()) as u64;
        let indirect_draw_in_bytes = std::mem::size_of_val(indirect_draw_data.as_slice()) as u64;
        let super_indirect_buffer: SubBuffer<u8, GpuIndirect> = SubBuffer::with_capacity(cmd_pool.device(),
                                                                                         indirect_dispatch_in_bytes +
                                                                                             indirect_draw_in_bytes)?;
        let offset = 0;
        let indirect_dispatch_buffer = super_indirect_buffer.sub(offset..offset + indirect_dispatch_in_bytes).reinterpret_into::<vk::DispatchIndirectCommand>();
        let offset = offset + indirect_dispatch_in_bytes;
        let indirect_draw_buffer = super_indirect_buffer.sub(offset..offset + indirect_draw_in_bytes).reinterpret_into::<vk::DrawIndirectCommand>();
        let offset = offset + indirect_draw_in_bytes;

        let indirect_dispatch = StageBuffer::wrap(cmd_pool, &indirect_dispatch_data, indirect_dispatch_buffer)?;
        let indirect_draw = StageBuffer::wrap(cmd_pool, &indirect_draw_data, indirect_draw_buffer)?;

        let indirect = Indirect::new(super_indirect_buffer, &indirect_dispatch, &indirect_draw);
        let player_event_uniform = HostBuffer::new(cmd_pool.device(), &[PlayerEvent::nothing()])?;

        let max_subgroup_size = cmd_pool.device().get_max_subgroup_size();
        println!("MAX subgroup size={}",max_subgroup_size);
        let mut specialization_constants = SpecializationConstants::new();
        specialization_constants.entry_uint(1,max_subgroup_size);//GROUP_SIZE
        specialization_constants.entry_uint(2,cap.world_size.width() as u32);//CHUNKS_X
        specialization_constants.entry_uint(3,cap.world_size.depth() as u32);//CHUNKS_Z
        specialization_constants.entry_uint(4,cap.faces_to_be_inserted_chunk_capacity);//
        specialization_constants.entry_uint(5,cap.faces_to_be_removed_chunk_capacity);//
        specialization_constants.entry_uint(6,2);//BROAD_PHASE_CELL_SIZE
        specialization_constants.entry_uint(7,8);//BROAD_PHASE_CELL_CAPACITY
        specialization_constants.entry_float(100,0.7);//BLOCK_COLLISION_FRICTION
        specialization_constants.entry_float(101,0.01);//BLOCK_COLLISION_MINIMUM_BOUNCE
        specialization_constants.entry_float(102,1.0);//PHYSICS_SIMULATION_DELTA_TIME_PER_STEP
        specialization_constants.entry_float(103,0.01);//BONE_COLLISION_FORCE_PER_AREA_UNIT
        specialization_constants.entry_float(104,0.2);//IMPULSE_AVERAGING_OVER_TIMESETP
        specialization_constants.entry_float(105,0f32);//GRAVITY
        specialization_constants.entry_float(106,0.99);//DAMPING_COEFFICIENT

        specialization_constants.entry_uint(300,cap.max_bones as u32);//MAX_BONES
        specialization_constants.entry_uint(301,cap.max_sensors as u32);//MAX_SENSORS
        specialization_constants.entry_uint(302,cap.max_faces as u32);//MAX_FACES
        specialization_constants.entry_uint(303,cap.max_persistent_floats as u32);//MAX_PERSISTENT_FLOATS
        specialization_constants.entry_uint(304,cap.max_neural_net_layers as u32);//MAX_NEURAL_NET_LAYERS
        specialization_constants.entry_uint(305,cap.max_tmp_faces_copy as u32);//MAX_TMP_FACES_COPY
        specialization_constants.entry_uint(306,cap.max_world_blocks_to_update as u32);//MAX_WORLD_BLOCKS_TO_UPDATE
        specialization_constants.entry_uint(307,cap.max_blocks_to_be_inserted_or_removed as u32);//MAX_BLOCKS_TO_BE_INSERTED_OR_REMOVED
        specialization_constants.entry_uint(308,cap.max_faces_to_be_inserted as u32);//MAX_FACES_TO_BE_INSERTED
        specialization_constants.entry_uint(309,cap.max_faces_to_be_removed as u32);//MAX_FACES_TO_BE_REMOVED

        Ok(Self {
            specialization_constants,
            face_count_per_chunk_buffer,
            opaque_and_transparent_face_buffer,
            faces,
            world_copy,
            world,
            world_size: cap.world_size.clone(),
            sampler,
            world_blocks_to_update: world_blocks_to_update_buffer,
            world_blocks_to_update_copy: world_blocks_to_update_copy_buffer,
            player_event_uniform,
            persistent_floats,
            neural_net_layers,
            collision_grid: grid_buffer,
            global_mutables,
            indirect_dispatch,
            indirect_draw,
            tmp_faces_copy: tmp_faces_copy_buffer,
            blocks_to_be_inserted_or_removed: blocks_to_be_inserted_or_removed_buffer,
            faces_to_be_inserted: faces_to_be_inserted_buffer,
            faces_to_be_removed: faces_to_be_removed_buffer,
            indirect,
            bones,
        })
    }
    pub fn build(self) -> Result<Foundations, Error> {
        let Self {
            specialization_constants,
            face_count_per_chunk_buffer,
            opaque_and_transparent_face_buffer,
            faces,
            world_copy,
            faces_to_be_inserted,
            faces_to_be_removed,
            tmp_faces_copy,
            blocks_to_be_inserted_or_removed,
            player_event_uniform,
            world_blocks_to_update,
            world_blocks_to_update_copy,
            indirect_dispatch,
            indirect_draw,
            world_size,
            world,
            bones,
            persistent_floats,
            collision_grid,
            global_mutables,
            indirect,
            neural_net_layers,
            sampler,
        } = self;
        let bones = bones.take()?.take_gpu();
        let neural_net_layers = neural_net_layers.take()?.take_gpu();
        let global_mutables = global_mutables.take()?.take_gpu();
        let _ = indirect_dispatch.take()?.take_gpu();
        let _ = indirect_draw.take()?.take_gpu();
        let faces = faces.take()?.take_gpu();
        let world_copy = world_copy.take()?.take_gpu();
        let world = world.take()?.take_gpu(); //wait for completion and then dispose of the staging buffer
        Ok(Foundations {
            specialization_constants,
            faces_to_be_inserted,
            faces_to_be_removed,
            tmp_faces_copy,
            blocks_to_be_inserted_or_removed,
            player_event_uniform,
            face_count_per_chunk_buffer,
            opaque_and_transparent_face_buffer,
            faces,
            world_copy,
            world_blocks_to_update,
            persistent_floats,
            world_blocks_to_update_copy,
            world,
            world_size,
            bones,
            collision_grid,
            neural_net_layers,
            global_mutables,
            indirect,
            sampler,
        })
    }
}

pub struct Foundations {
    specialization_constants: SpecializationConstants,
    faces: SubBuffer<Face, Storage>,
    face_count_per_chunk_buffer: SubBuffer<Face, Storage>,
    opaque_and_transparent_face_buffer: SubBuffer<Face, Storage>,
    world_copy: SubBuffer<Block, Storage>,
    faces_to_be_inserted: SubBuffer<Face, Storage>,
    faces_to_be_removed: SubBuffer<u32, Storage>,
    world_size: WorldSize,
    tmp_faces_copy: SubBuffer<u32, Storage>,
    blocks_to_be_inserted_or_removed: SubBuffer<u32, Storage>,
    player_event_uniform: HostBuffer<PlayerEvent, Uniform>,
    world_blocks_to_update: SubBuffer<u32, Storage>,
    world_blocks_to_update_copy: SubBuffer<u32, Storage>,
    world: SubBuffer<Block, Storage>,
    bones: SubBuffer<Bone, Storage>,
    neural_net_layers: SubBuffer<NeuralNetLayer, Storage>,
    persistent_floats: SubBuffer<f32, Storage>,
    global_mutables: SubBuffer<GlobalMutables, Storage>,
    collision_grid: SubBuffer<u32, Storage>,
    indirect: Indirect,
    sampler: Sampler,
}

impl Foundations {
    pub fn specialization_constants(&self) -> &SpecializationConstants{
        &self.specialization_constants
    }
    pub fn tmp_faces_copy(&self) -> &SubBuffer<u32, Storage> {
        &self.tmp_faces_copy
    }
    pub fn faces(&self) -> &SubBuffer<Face, Storage> {
        &self.faces
    }
    pub fn face_count_per_chunk_buffer(&self) -> &SubBuffer<Face, Storage> {
        &self.face_count_per_chunk_buffer
    }
    pub fn opaque_and_transparent_face_buffer(&self) -> &SubBuffer<Face, Storage> {
        &self.opaque_and_transparent_face_buffer
    }
    pub fn player_event_uniform(&self) -> &HostBuffer<PlayerEvent, Uniform> {
        &self.player_event_uniform
    }
    pub fn player_event_uniform_mut(&mut self) -> &mut HostBuffer<PlayerEvent, Uniform> {
        &mut self.player_event_uniform
    }
    pub fn world_size(&self) -> &WorldSize {
        &self.world_size
    }
    pub fn indirect(&self) -> &Indirect {
        &self.indirect
    }
    pub fn bones(&self) -> &SubBuffer<Bone, Storage> {
        &self.bones
    }
    pub fn world_blocks_to_update(&self) -> &SubBuffer<u32, Storage> {
        &self.world_blocks_to_update
    }
    pub fn faces_to_be_inserted(&self) -> &SubBuffer<Face, Storage> {
        &self.faces_to_be_inserted
    }
    pub fn faces_to_be_removed(&self) -> &SubBuffer<u32, Storage> {
        &self.faces_to_be_removed
    }
    pub fn world_blocks_to_update_copy(&self) -> &SubBuffer<u32, Storage> {
        &self.world_blocks_to_update_copy
    }
    pub fn blocks_to_be_inserted_or_removed(&self) -> &SubBuffer<u32, Storage> {
        &self.blocks_to_be_inserted_or_removed
    }
    pub fn persistent_floats(&self) -> &SubBuffer<f32, Storage> {
        &self.persistent_floats
    }
    pub fn neural_net_layers(&self) -> &SubBuffer<NeuralNetLayer, Storage> {
        &self.neural_net_layers
    }
    pub fn world(&self) -> &SubBuffer<Block, Storage> {
        &self.world
    }
    pub fn world_copy(&self) -> &SubBuffer<Block, Storage> {
        &self.world_copy
    }
    pub fn global_mutables(&self) -> &SubBuffer<GlobalMutables, Storage> {
        &self.global_mutables
    }
    pub fn collision_grid(&self) -> &SubBuffer<u32, Storage> {
        &self.collision_grid
    }
}
