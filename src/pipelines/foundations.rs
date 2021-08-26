use crate::render::stage_buffer::{StageBuffer, StageSubBuffer, IndirectDispatchSubBuffer, IndirectSubBuffer};
use crate::pipelines::particle::Particle;
use crate::render::command_pool::{CommandPool};


use ash::vk;


use failure::Error;


use crate::render::submitter::{Submitter, fill_submit};

use crate::render::buffer_type::{Cpu, Storage, GpuIndirect};

use crate::blocks::world_size::CHUNK_VOLUME_IN_CELLS;
use crate::render::subbuffer::SubBuffer;
use crate::pipelines::constraint::Constraint;
use crate::render::buffer::Buffer;
use crate::pipelines::particle_constants::ParticleConstants;
use crate::blocks::{WorldSize, Block, Face, WorldBlocks, WorldFaces};
use crate::render::sampler::Sampler;
use crate::pipelines::bone::Bone;
use crate::blocks::block_properties::{BLOCKS, BlockProp, BEDROCK, DIRT, GRASS, GLASS, PLANK, AIR};
use crate::pipelines::sensor::Sensor;
use crate::pipelines::neural_net_layer::NeuralNetLayer;
use crate::pipelines::neural_net_layer::Aggregate::Overwrite;
use crate::pipelines::muscle::Muscle;

pub struct Indirect {
    collision_detection: SubBuffer<vk::DispatchIndirectCommand, GpuIndirect>,
    solve_constraints: SubBuffer<vk::DispatchIndirectCommand, GpuIndirect>,
    update_bones: SubBuffer<vk::DispatchIndirectCommand, GpuIndirect>,
    agent_sensory_input_update: SubBuffer<vk::DispatchIndirectCommand, GpuIndirect>,
    feed_forward_net: SubBuffer<vk::DispatchIndirectCommand, GpuIndirect>,
    draw_bones: SubBuffer<vk::DrawIndirectCommand, GpuIndirect>,
    draw_blocks: SubBuffer<vk::DrawIndirectCommand, GpuIndirect>,
    draw_particles: SubBuffer<vk::DrawIndirectCommand, GpuIndirect>,
    super_indirect_buffer: SubBuffer<u8, GpuIndirect>
}

impl Indirect {
    fn new(super_indirect_buffer: SubBuffer<u8, GpuIndirect>, indirect_dispatch: &Submitter<IndirectDispatchSubBuffer>, indirect_draw: &Submitter<IndirectSubBuffer>) -> Self {
        let collision_detection = indirect_dispatch.gpu().element(0);
        let solve_constraints = indirect_dispatch.gpu().element(1);
        let update_bones = indirect_dispatch.gpu().element(2);
        let agent_sensory_input_update = indirect_dispatch.gpu().element(3);
        let feed_forward_net = indirect_dispatch.gpu().element(4);
        let draw_bones = indirect_draw.gpu().element(0);
        let draw_blocks = indirect_draw.gpu().element(1);
        let draw_particles = indirect_draw.gpu().element(2);
        Self {
            super_indirect_buffer,
            collision_detection,
            solve_constraints,
            update_bones,
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
    pub fn collision_detection(&self) -> &SubBuffer<vk::DispatchIndirectCommand, GpuIndirect> {
        &self.collision_detection
    }
    pub fn feed_forward_net(&self) -> &SubBuffer<vk::DispatchIndirectCommand, GpuIndirect> {
        &self.feed_forward_net
    }
    pub fn agent_sensory_input_update(&self) -> &SubBuffer<vk::DispatchIndirectCommand, GpuIndirect> {
        &self.agent_sensory_input_update
    }
    pub fn solve_constraints(&self) -> &SubBuffer<vk::DispatchIndirectCommand, GpuIndirect> {
        &self.solve_constraints
    }
    pub fn update_bones(&self) -> &SubBuffer<vk::DispatchIndirectCommand, GpuIndirect> {
        &self.update_bones
    }
}

pub struct FoundationInitializer {
    particles: Submitter<StageSubBuffer<Particle, Cpu, Storage>>,
    world: Submitter<StageSubBuffer<Block,Cpu,Storage>>,
    faces: Submitter<StageSubBuffer<Face,Cpu,Storage>>,
    face_count_per_chunk_buffer:SubBuffer<Face, Storage>,
    opaque_and_transparent_face_buffer:SubBuffer<Face, Storage>,
    collision_grid: Submitter<SubBuffer<u32, Storage>>,
    constraints: Submitter<StageSubBuffer<Constraint, Cpu, Storage>>,
    block_properties: Submitter<StageSubBuffer<BlockProp, Cpu, Storage>>,
    bones: Submitter<StageSubBuffer<Bone, Cpu, Storage>>,
    sensors: Submitter<StageSubBuffer<Sensor, Cpu, Storage>>,
    muscles: Submitter<StageSubBuffer<Muscle, Cpu, Storage>>,
    persistent_floats: Submitter<StageSubBuffer<f32, Cpu, Storage>>,
    neural_net_layers: Submitter<StageSubBuffer<NeuralNetLayer, Cpu, Storage>>,
    particle_constants: Submitter<StageSubBuffer<ParticleConstants, Cpu, Storage>>,
    indirect_dispatch: Submitter<IndirectDispatchSubBuffer>,
    indirect_draw: Submitter<IndirectSubBuffer>,
    indirect: Indirect,
    sampler: Sampler,
    world_size: WorldSize,
}

impl FoundationInitializer {
    pub fn particle_constants(&self) -> &StageSubBuffer<ParticleConstants, Cpu, Storage>{
        &self.particle_constants
    }
    pub fn face_count_per_chunk_buffer(&self)->&SubBuffer<Face, Storage>{
        &self.face_count_per_chunk_buffer
    }
    pub fn opaque_and_transparent_face_buffer(&self)->&SubBuffer<Face, Storage>{
        &self.opaque_and_transparent_face_buffer
    }
    pub fn sensors(&self) -> &StageSubBuffer<Sensor, Cpu,Storage> {
        &self.sensors
    }
    pub fn persistent_floats(&self) -> &StageSubBuffer<f32,Cpu, Storage> {
        &self.persistent_floats
    }
    pub fn indirect(&self) -> &Indirect {
        &self.indirect
    }
    pub fn particles(&self) -> &StageSubBuffer<Particle, Cpu, Storage> {
        &self.particles
    }
    pub fn constraints(&self) -> &StageSubBuffer<Constraint, Cpu, Storage> {
        &self.constraints
    }
    pub fn collision_grid(&self) -> &SubBuffer<u32, Storage> {
        &self.collision_grid
    }
    pub fn bones(&self) -> &StageSubBuffer<Bone, Cpu, Storage> {
        &self.bones
    }
    pub fn sampler(&self) -> &Sampler {
        &self.sampler
    }
    pub fn world_size(&self) -> &WorldSize {
        &self.world_size
    }
    pub fn world(&self) -> &StageSubBuffer<Block, Cpu,Storage> {
        &self.world
    }
    pub fn faces(&self) -> &StageSubBuffer<Face, Cpu,Storage> {
        &self.faces
    }
    pub fn muscles(&self) -> &StageSubBuffer<Muscle, Cpu,Storage> {
        &self.muscles
    }

    pub fn block_properties(&self) -> &StageSubBuffer<BlockProp, Cpu, Storage> {
        &self.block_properties
    }
    pub fn new(cmd_pool: &CommandPool) -> Result<Self, failure::Error> {
        let world_size = WorldSize::new(2,2);
        let particles = 512u64;
        let bones = 128u64;
        let faces = 2048u64*world_size.total_chunks() as u64;
        let max_constraints = 128u64;
        let grid_size = CHUNK_VOLUME_IN_CELLS as u64; // this can also be reinterpreted as extra
        // non-persistent backing memory for auxiliary matrices and tensors used by neural networks
        // for carrying out their computations
        let solid_particles = 256;
        let phantom_particles = 256;
        let sensors = 128u64;
        let persistent_floats = 512u64; // used as backing memory for vectors, matrices and
        // tensors that make up various neural networks. Especially, the outputs of recursive neural networks
        // often need to be persistent, because those outputs are later fed as inputs to the same neural net.
        let neural_net_layers = 128u64;
        let muscles = 128u64;
        debug_assert!(solid_particles + phantom_particles <= particles);

        let mut world_blocks = WorldBlocks::new(world_size.clone());
        world_blocks.no_update_fill_level(0, 1, BEDROCK);
        world_blocks.no_update_fill_level(1, 1, DIRT);
        world_blocks.no_update_fill_level(2, 1, GRASS);
        world_blocks.no_update_fill_level(10, 1, GLASS);
        world_blocks.no_update_outline(5, 2, 5, 5, 5, 5, PLANK);
        // world_blocks.no_update_set_block(1,2,1,AIR);
        let mut world_faces = WorldFaces::with_capacity(world_size.clone(), faces as usize);
        world_faces.generate_faces(&world_blocks);
        let w2 = 0.4f32;
        let w = w2/2.;
        let h2 = 0.4f32;
        let h = h2/2.;
        let l2 = 0.6f32;
        let l = l2/2.;
        let s = 0.2f32;
        let diag_w2_h2 = (w2 * w2 + h2 * h2).sqrt();
        let diag_w2_l2 = (w2 * w2 + l2 * l2).sqrt();
        let diag_s_l2 = (s*s+ l2 * l2).sqrt();
        let diag_w_l2 = (w*w+ l2 * l2).sqrt();
        let diag_s_l = (s*s+ l * l).sqrt();
        let diag_w_l = (w*w+ l * l).sqrt();
        let depth = 0.1;
        let diag_depth_half_w = (w*w/4.+depth*depth).sqrt();
        let diag_depth_l_half_w = (w*w/4.+l*l+depth*depth).sqrt();
        let diag_depth_l_w = (w*w+l*l+depth*depth).sqrt();
        let mut particles_data: Vec<Particle> = std::iter::repeat_with(Particle::random).take((solid_particles+phantom_particles) as usize).collect();
        particles_data[1].new_position = glm::vec3(2., 7., 2.);
        particles_data[1].old_position = particles_data[1].new_position;
        particles_data[2].new_position = particles_data[1].new_position + glm::vec3(w2, 0., 0.);
        particles_data[2].old_position = particles_data[2].new_position;
        particles_data[3].new_position = particles_data[1].new_position + glm::vec3(w2, l2, 0.);
        particles_data[3].old_position = particles_data[3].new_position;
        particles_data[4].new_position = particles_data[1].new_position + glm::vec3(0., l2, 0.);
        particles_data[4].old_position = particles_data[4].new_position;
        particles_data[5].new_position = particles_data[1].new_position + glm::vec3(w2, h2 + l2, 0.);
        particles_data[5].old_position = particles_data[5].new_position;
        particles_data[6].new_position = particles_data[1].new_position + glm::vec3(0., h2 + l2, 0.);
        particles_data[6].old_position = particles_data[6].new_position;
        particles_data[7].new_position = particles_data[1].new_position + glm::vec3(w2 +s, l2, 0.);
        particles_data[7].old_position = particles_data[7].new_position;
        particles_data[8].new_position = particles_data[1].new_position + glm::vec3(-s, l2, 0.);
        particles_data[8].old_position = particles_data[8].new_position;
        particles_data[9].new_position = particles_data[1].new_position + glm::vec3(-s, 0., 0.);
        particles_data[9].old_position = particles_data[9].new_position;
        particles_data[10].new_position = particles_data[1].new_position + glm::vec3(w2 +s, 0., 0.);
        particles_data[10].old_position = particles_data[10].new_position;
        particles_data[11].new_position = particles_data[1].new_position + glm::vec3(0., -l2, 0.);
        particles_data[11].old_position = particles_data[11].new_position;
        particles_data[12].new_position = particles_data[1].new_position + glm::vec3(w2, -l2, 0.);
        particles_data[12].old_position = particles_data[12].new_position;
        particles_data[13].new_position = particles_data[1].new_position + glm::vec3(w, 0., 0.);
        particles_data[13].old_position = particles_data[13].new_position;
        particles_data[14].new_position = particles_data[1].new_position + glm::vec3(-s, l, 0.);
        particles_data[14].old_position = particles_data[14].new_position;
        particles_data[15].new_position = particles_data[1].new_position + glm::vec3(w2+s, l, 0.);
        particles_data[15].old_position = particles_data[15].new_position;
        particles_data[16].new_position = particles_data[1].new_position + glm::vec3(0., -l, 0.);
        particles_data[16].old_position = particles_data[16].new_position;
        particles_data[17].new_position = particles_data[1].new_position + glm::vec3(w, -l, 0.);
        particles_data[17].old_position = particles_data[17].new_position;
        particles_data[18].new_position = particles_data[1].new_position + glm::vec3(w, l, depth);
        particles_data[18].old_position = particles_data[18].new_position;
        particles_data[solid_particles as usize+0].new_position = particles_data[1].new_position + glm::vec3(0., 0., 0.);
        particles_data[solid_particles as usize+0].old_position = particles_data[solid_particles as usize+0].new_position;
        particles_data[solid_particles as usize+1].new_position = particles_data[1].new_position + glm::vec3(w2, 0., 0.);
        particles_data[solid_particles as usize+1].old_position = particles_data[solid_particles as usize +1].new_position;
        particles_data[solid_particles as usize+2].new_position = particles_data[1].new_position + glm::vec3(w, -l2, 0.);
        particles_data[solid_particles as usize+2].old_position = particles_data[solid_particles as usize +2].new_position;
        particles_data[solid_particles as usize+3].new_position = particles_data[1].new_position + glm::vec3(w, -l2, 0.);
        particles_data[solid_particles as usize+3].old_position = particles_data[solid_particles as usize +3].new_position;
        particles_data[solid_particles as usize+4].new_position = particles_data[1].new_position + glm::vec3(0., l, 0.);
        particles_data[solid_particles as usize+4].old_position = particles_data[solid_particles as usize +4].new_position;
        particles_data[solid_particles as usize+5].new_position = particles_data[1].new_position + glm::vec3(w2, l, 0.);
        particles_data[solid_particles as usize+5].old_position = particles_data[solid_particles as usize +5].new_position;
        particles_data[solid_particles as usize+6].new_position = particles_data[1].new_position + glm::vec3(w, -l, 0.);
        particles_data[solid_particles as usize+6].old_position = particles_data[solid_particles as usize +6].new_position;
        particles_data[solid_particles as usize+7].new_position = particles_data[1].new_position + glm::vec3(w, -l, 0.);
        particles_data[solid_particles as usize+7].old_position = particles_data[solid_particles as usize +7].new_position;
        particles_data[solid_particles as usize+8].new_position = particles_data[1].new_position + glm::vec3(w/2., -l2, -depth);
        particles_data[solid_particles as usize+8].old_position = particles_data[solid_particles as usize +8].new_position;
        particles_data[solid_particles as usize+9].new_position = particles_data[1].new_position + glm::vec3(w+w/2., -l2, -depth);
        particles_data[solid_particles as usize+9].old_position = particles_data[solid_particles as usize +9].new_position;
        let stiffness = 0.2f32;
        let mut predefined_constraints = vec![
            // Quad(1 2 3 4)
            Constraint::distance(stiffness,1, 2, w2),
            Constraint::distance(stiffness,2, 3, l2),
            Constraint::distance(stiffness,3, 4, w2),
            Constraint::distance(stiffness,4, 1, l2),
            Constraint::distance(stiffness,4, 2, diag_w2_l2),
            Constraint::distance(stiffness,1, 3, diag_w2_l2),
            // Quad(6 5 3 4)
            Constraint::distance(stiffness,6, 5, w2),
            Constraint::distance(stiffness,5, 3, h2),
            Constraint::distance(stiffness,6, 4, w2),
            Constraint::distance(stiffness,4, 5, diag_w2_h2),
            Constraint::distance(stiffness,6, 3, diag_w2_h2),
            // Quad(3 7 15 solid_particles+5)
            Constraint::distance(stiffness,3, 7, s),
            Constraint::distance(stiffness,7, 15, l),
            Constraint::distance(stiffness,15, solid_particles as u32+5, s),
            Constraint::distance(stiffness,solid_particles as u32+5, 3, l),
            Constraint::distance(stiffness,solid_particles as u32+5, 7, diag_s_l),
            Constraint::distance(stiffness,3, 15, diag_s_l),
            // Quad(solid_particles+5 15 10 solid_particles+1)
            Constraint::distance(stiffness,15, 10, l),
            Constraint::distance(stiffness,10, solid_particles as u32+1, s),
            Constraint::distance(stiffness,solid_particles as u32+1, solid_particles as u32+5, l),
            Constraint::distance(stiffness,solid_particles as u32+1, 15, diag_s_l),
            Constraint::distance(stiffness,solid_particles as u32+5, 10, diag_s_l),
            // Quad(4 8 14 solid_particles+4)
            Constraint::distance(stiffness,4, 8, s),
            Constraint::distance(stiffness,8, 14, l),
            Constraint::distance(stiffness,14, solid_particles as u32+4, s),
            Constraint::distance(stiffness,solid_particles as u32+4, 4, l),
            Constraint::distance(stiffness,solid_particles as u32+4, 8, diag_s_l),
            Constraint::distance(stiffness,14, 4, diag_s_l),
            // Quad(14 solid_particles+4 solid_particles+0 9)
            Constraint::distance(stiffness,solid_particles as u32+4, solid_particles as u32+0, l),
            Constraint::distance(stiffness,solid_particles as u32+0, 9, s),
            Constraint::distance(stiffness,9, 14, l),
            Constraint::distance(stiffness,solid_particles as u32+0, 14, diag_s_l),
            Constraint::distance(stiffness,9, solid_particles as u32+4, diag_s_l),
            // Quad(11 solid_particles+2 solid_particles+6 16)
            Constraint::distance(stiffness,11, solid_particles as u32+2,w),
            Constraint::distance(stiffness,solid_particles as u32+2, solid_particles as u32+6, l),
            Constraint::distance(stiffness,solid_particles as u32+6,16,w),
            Constraint::distance(stiffness,16, 11, l),
            Constraint::distance(stiffness,solid_particles as u32+2, 16, diag_w_l),
            Constraint::distance(stiffness,11, solid_particles as u32+6, diag_w_l),
            // Quad(16 solid_particles+6 13 1)
            Constraint::distance(stiffness,solid_particles as u32+6, 13, l),
            Constraint::distance(stiffness,13,1,w),
            Constraint::distance(stiffness,1, 16, l),
            Constraint::distance(stiffness,solid_particles as u32+6, 1, diag_w_l),
            Constraint::distance(stiffness,16, 13, diag_w_l),
            // Quad(solid_particles+3 12 17 solid_particles+7)
            Constraint::distance(stiffness,solid_particles as u32+3,12,w),
            Constraint::distance(stiffness,12, 17, l),
            Constraint::distance(stiffness,17,solid_particles as u32+7, w),
            Constraint::distance(stiffness,solid_particles as u32+7, solid_particles as u32+3, l),
            Constraint::distance(stiffness,17, solid_particles as u32+3, diag_w_l),
            Constraint::distance(stiffness,12, solid_particles as u32+7, diag_w_l),
            // Quad(17 solid_particles+7 13 2)
            Constraint::distance(stiffness,17, 2, l),
            Constraint::distance(stiffness,2,13, w),
            Constraint::distance(stiffness,13, solid_particles as u32+7, l),
            Constraint::distance(stiffness,2, solid_particles as u32+7, diag_w_l),
            Constraint::distance(stiffness,17, 13, diag_w_l),
            // Depth constraints
            Constraint::distance(stiffness, solid_particles as u32+8, 11, diag_depth_half_w),
            Constraint::distance(stiffness, solid_particles as u32+8, solid_particles as u32+2, diag_depth_half_w),
            Constraint::distance(stiffness,solid_particles as u32+8, 16, diag_depth_l_half_w),
            Constraint::distance(stiffness, solid_particles as u32+9, 12, diag_depth_half_w),
            Constraint::distance(stiffness, solid_particles as u32+9, solid_particles as u32+3, diag_depth_half_w),
            Constraint::distance(stiffness,solid_particles as u32+9, 17, diag_depth_l_half_w),
            Constraint::distance(stiffness,18, 1, diag_depth_l_w),
            Constraint::distance(stiffness,18, 2, diag_depth_l_w),
            Constraint::distance(stiffness,18, 3, diag_depth_l_w),
            Constraint::distance(stiffness,18, 4, diag_depth_l_w),
        ];
        let muscle_constraints_offset = predefined_constraints.len();
        predefined_constraints.append(&mut vec![
            Constraint::distance(stiffness,solid_particles as u32+8, 1, w),
            Constraint::distance(stiffness,solid_particles as u32+9, 2, w),
            Constraint::distance(stiffness,solid_particles as u32+6, 18, l),
            Constraint::distance(stiffness,solid_particles as u32+7, 18, l),
            Constraint::distance(stiffness,solid_particles as u32+5, 18, l),
            Constraint::distance(stiffness,solid_particles as u32+4, 18, l),
            Constraint::distance(stiffness,solid_particles as u32+1, 7, diag_s_l2),
            Constraint::distance(stiffness,solid_particles as u32+0, 8, diag_s_l2),
        ]);
        let muscle_constraints_length = predefined_constraints.len() - muscle_constraints_offset;

        let block_properties_data:Vec<BlockProp> = BLOCKS.iter().map(|p|p.prop).collect();

        let bone_data = vec![
            Bone::new([11,solid_particles as u32+2, solid_particles as u32+6, 16], 0, 0.1),
            Bone::new([16,solid_particles as u32+6, 13, 1], 1, 0.1),
            Bone::new([solid_particles as u32+3, 12,17,solid_particles as u32+7], 2, 0.1),
            Bone::new([solid_particles as u32+7, 17,2,13], 3, 0.1),
            Bone::new([solid_particles as u32+1, 10, 15, solid_particles as u32+5], 4, 0.1),
            Bone::new([solid_particles as u32+5, 15, 7, 3], 5, 0.1),
            Bone::new([9,solid_particles as u32+0, solid_particles as u32+4, 14], 6, 0.1),
            Bone::new([14, solid_particles as u32+4, 4, 8], 7, 0.1),
            Bone::new([1, 2, 3, 4], 8, 0.1),
            Bone::new([4, 3, 5, 6], 9, 0.2),
        ];

        let sensor_data = vec![
            Sensor::new_movement_sensor(1),
            Sensor::new_movement_sensor(2),
            Sensor::new_movement_sensor(4),
            Sensor::new_movement_sensor(5),
            Sensor::new_movement_sensor(6),
            Sensor::new_movement_sensor(7),
            Sensor::new_movement_sensor(8),
            Sensor::new_movement_sensor(9),
            Sensor::new_movement_sensor(10),
            Sensor::new_movement_sensor(11),
            Sensor::new_movement_sensor(12),
            Sensor::new_movement_sensor(16),
            Sensor::new_movement_sensor(17),
        ];
        let muscle_data = vec![
            Muscle::new(muscle_constraints_offset as u32,w2,l2),
            Muscle::new(muscle_constraints_offset as u32+1,w2,l2),
            Muscle::new(muscle_constraints_offset as u32+2,w2,l2),
            Muscle::new(muscle_constraints_offset as u32+3,w2,l2),
            Muscle::new(muscle_constraints_offset as u32+4,w,l+w),
            Muscle::new(muscle_constraints_offset as u32+5,w,l+w),
            Muscle::new(muscle_constraints_offset as u32+6,w,diag_s_l2),
            Muscle::new(muscle_constraints_offset as u32+7,w,diag_s_l2),
        ];
        let persistent_floats_data = (0..persistent_floats).map(|_|rand::random::<f32>()).collect::<Vec<f32>>();

        let constants = ParticleConstants {
            predefined_constraints: predefined_constraints.len() as i32,
            collision_constraints: 0,
            solid_particles: solid_particles as i32,
            phantom_particles: phantom_particles as i32,
            chunks_x: world_size.width() as i32,
            chunks_z: world_size.depth() as i32,
            bones: bone_data.len() as i32,
            world_width: world_size.world_width()as i32,
            world_depth: world_size.world_depth() as i32,
            world_area: world_size.world_area()as i32,
            total_chunks: world_size.total_chunks()as i32,
            sensors: sensor_data.len() as u32
        };
        let sensor_input_length = 13*3;
        let recurrent_length = 13;
        let weights_offset = 32;
        let neural_net_layer_data = vec![
            NeuralNetLayer::new_input_recurrent(0, sensor_input_length,weights_offset,recurrent_length ),
            NeuralNetLayer::new_hidden(0,sensor_input_length, weights_offset+recurrent_length, sensor_input_length,recurrent_length+muscle_constraints_length as u32,None,Overwrite),
            NeuralNetLayer::new_output(sensor_input_length,0, muscle_constraints_length as u32, 0,weights_offset ,recurrent_length),
        ];

        let particles_in_bytes = std::mem::size_of::<Particle>() as u64 * particles;
        let faces_in_bytes = std::mem::size_of::<Face>() as u64 * faces;
        let grid_in_bytes = std::mem::size_of::<u32>() as u64 * grid_size;
        let block_properties_in_bytes = std::mem::size_of_val(block_properties_data.as_slice()) as u64;
        let world_in_bytes = (std::mem::size_of::<Block>()*world_size.world_volume()) as u64;
        let constraints_in_bytes = std::mem::size_of::<Constraint>() as u64 * max_constraints;
        let bones_in_bytes = std::mem::size_of::<Bone>() as u64 * bones;
        let constants_in_bytes = std::mem::size_of_val(&constants) as u64;
        let sensors_in_bytes = std::mem::size_of::<Sensor>() as u64 * sensors;
        let persistent_floats_in_bytes = std::mem::size_of::<f32>() as u64 * persistent_floats;
        let neural_net_layers_in_bytes = std::mem::size_of::<NeuralNetLayer>() as u64 * neural_net_layers;
        let muscles_in_bytes = std::mem::size_of::<Muscle>() as u64 * muscles;

        let super_buffer: SubBuffer<u8, Storage> = SubBuffer::with_capacity(cmd_pool.device(),
                                                                            particles_in_bytes +
                                                                                faces_in_bytes +
                                                                                grid_in_bytes +
                                                                                block_properties_in_bytes +
                                                                                world_in_bytes +
                                                                                constraints_in_bytes +
                                                                                bones_in_bytes +
                                                                                constants_in_bytes +
                                                                                sensors_in_bytes +
                                                                                persistent_floats_in_bytes +
                                                                                neural_net_layers_in_bytes +
                                                                                muscles_in_bytes)?;
        let offset = 0;
        let particle_buffer = super_buffer.sub(offset..offset + particles_in_bytes).reinterpret_into::<Particle>();
        let offset = offset + particles_in_bytes;
        assert_eq!(offset%16,0); // check correct GLSL alignment
        let face_buffer = super_buffer.sub(offset..offset + faces_in_bytes).reinterpret_into::<Face>();
        let offset = offset + faces_in_bytes;
        assert_eq!(offset%16,0);
        let grid_buffer = super_buffer.sub(offset..offset + grid_in_bytes).reinterpret_into::<u32>();
        let offset = offset + grid_in_bytes;
        assert_eq!(offset%16,0);
        let block_properties_buffer = super_buffer.sub(offset..offset + block_properties_in_bytes).reinterpret_into::<BlockProp>();
        let offset = offset + block_properties_in_bytes;
        assert_eq!(offset%16,0);
        let world_buffer = super_buffer.sub(offset..offset + world_in_bytes).reinterpret_into::<Block>();
        let offset = offset + world_in_bytes;
        assert_eq!(offset%16,0);
        let constraint_buffer = super_buffer.sub(offset..offset + constraints_in_bytes).reinterpret_into::<Constraint>();
        let offset = offset + constraints_in_bytes;
        assert_eq!(offset%16,0);
        let bones_buffer = super_buffer.sub(offset..offset + bones_in_bytes).reinterpret_into::<Bone>();
        let offset = offset + bones_in_bytes;
        assert_eq!(offset%16,0);
        let constants_buffer = super_buffer.sub(offset..offset + constants_in_bytes).reinterpret_into::<ParticleConstants>();
        let offset = offset + constants_in_bytes;
        assert_eq!(offset%16,0);
        let sensors_buffer = super_buffer.sub(offset..offset + sensors_in_bytes).reinterpret_into::<Sensor>();
        let offset = offset + sensors_in_bytes;
        assert_eq!(offset%16,0);
        let persistent_floats_buffer = super_buffer.sub(offset..offset + persistent_floats_in_bytes).reinterpret_into::<f32>();
        let offset = offset + persistent_floats_in_bytes;
        assert_eq!(offset%16,0);
        let neural_net_layers_buffer = super_buffer.sub(offset..offset + neural_net_layers_in_bytes).reinterpret_into::<NeuralNetLayer>();
        let offset = offset + neural_net_layers_in_bytes;
        assert_eq!(offset%16,0);
        let muscles_buffer = super_buffer.sub(offset..offset + muscles_in_bytes).reinterpret_into::<Muscle>();
        let offset = offset + muscles_in_bytes;
        assert_eq!(offset%16,0);

        let particle_constants = StageBuffer::wrap(cmd_pool, &[constants], constants_buffer)?;

        let particles = StageBuffer::wrap(cmd_pool, &particles_data, particle_buffer)?;

        let face_count_per_chunk_buffer = face_buffer.sub(..std::mem::size_of::<Face>() as u64 *world_size.total_chunks() as u64*2);
        let opaque_and_transparent_face_buffer = face_buffer.sub(std::mem::size_of::<Face>() as u64  * world_size.total_chunks() as u64*2..);
        let faces = StageBuffer::wrap(cmd_pool,  world_faces.as_slice(), face_buffer)?;

        let mut collision_grid = Submitter::new(grid_buffer, cmd_pool)?;
        fill_submit(&mut collision_grid, u32::MAX)?;

        let block_properties = StageBuffer::wrap(cmd_pool, &block_properties_data, block_properties_buffer)?;

        let bones = StageBuffer::wrap(cmd_pool, &bone_data, bones_buffer)?;

        let sensors = StageBuffer::wrap(cmd_pool, &sensor_data, sensors_buffer)?;

        let persistent_floats = StageBuffer::wrap(cmd_pool, &persistent_floats_data, persistent_floats_buffer)?;

        let world = StageBuffer::wrap(cmd_pool,  world_blocks.as_slice(), world_buffer)?;

        let constraints = StageBuffer::wrap(cmd_pool, &predefined_constraints, constraint_buffer)?;

        let neural_net_layers = StageBuffer::wrap(cmd_pool, &neural_net_layer_data, neural_net_layers_buffer)?;

        let muscles = StageBuffer::wrap(cmd_pool, &muscle_data, muscles_buffer)?;

        let sampler = Sampler::new(cmd_pool.device(), vk::Filter::NEAREST, true)?;

        fn dispatch_indirect(x: usize) -> vk::DispatchIndirectCommand {
            vk::DispatchIndirectCommand {
                x: (x as f32/ 32.).ceil() as u32,
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
            dispatch_indirect(phantom_particles.max(solid_particles) as usize ),// collision_detection.comp
            dispatch_indirect(0), // solve_constraints.comp
            dispatch_indirect(bone_data.len() ), // update_bones.comp
            dispatch_indirect(sensor_data.len() ), // agent_sensory_input_update.comp
            vk::DispatchIndirectCommand {  // feed_forward_net.comp
                x: (neural_net_layer_data.len()/3) as u32,
                y: 1,
                z: 1,
            }
        ];
        let indirect_draw_data = vec![
            draw_indirect(36, bone_data.len() as u32),// bones.vert
            draw_indirect(6, world_faces.len() as u32),// block.vert
            draw_indirect((solid_particles+phantom_particles) as u32, 1),// particles.vert
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

        Ok(Self {
            face_count_per_chunk_buffer,
            opaque_and_transparent_face_buffer,
            block_properties,
            faces,
            world,
            world_size,
            sampler,
            particles,
            constraints,
            sensors,
            persistent_floats,
            neural_net_layers,
            collision_grid,
            particle_constants,
            indirect_dispatch,
            indirect_draw,
            muscles,
            indirect,
            bones,
        })
    }
    pub fn build(self) -> Result<Foundations, Error> {
        let Self {
            face_count_per_chunk_buffer,
            opaque_and_transparent_face_buffer,
            block_properties,
            indirect_dispatch,
            indirect_draw,
            world_size,
            faces,
            world,
            bones,
            sensors,
            persistent_floats,
            particles,
            collision_grid,
            constraints,
            particle_constants,
            indirect,
            neural_net_layers,
            muscles,
            sampler,
        } = self;
        let particles = particles.take()?.take_gpu();
        let collision_grid = collision_grid.take()?;
        let block_properties = block_properties.take()?.take_gpu();
        let constraints = constraints.take()?.take_gpu();
        let bones = bones.take()?.take_gpu();
        let sensors = sensors.take()?.take_gpu();
        let muscles = muscles.take()?.take_gpu();
        let neural_net_layers = neural_net_layers.take()?.take_gpu();
        let persistent_floats = persistent_floats.take()?.take_gpu();
        let particle_constants = particle_constants.take()?.take_gpu();
        let _ = indirect_dispatch.take()?.take_gpu();
        let _ = indirect_draw.take()?.take_gpu();
        let world = world.take()?.take_gpu(); //wait for completion and then dispose of the staging buffer
        let faces = faces.take()?.take_gpu();
        Ok(Foundations {
            muscles,
            face_count_per_chunk_buffer,
            opaque_and_transparent_face_buffer,
            block_properties,
            faces,
            sensors,
            persistent_floats,
            world,
            world_size,
            bones,
            collision_grid,
            neural_net_layers,
            particle_constants,
            constraints,
            particles,
            indirect,
            sampler,
        })
    }
}

pub struct Foundations {
    world_size: WorldSize,
    block_properties: SubBuffer<BlockProp, Storage>,
    faces: SubBuffer<Face, Storage>,
    face_count_per_chunk_buffer: SubBuffer<Face, Storage>,
    opaque_and_transparent_face_buffer: SubBuffer<Face, Storage>,
    world: SubBuffer<Block, Storage>,
    particles: SubBuffer<Particle, Storage>,
    constraints: SubBuffer<Constraint, Storage>,
    bones: SubBuffer<Bone, Storage>,
    neural_net_layers: SubBuffer<NeuralNetLayer, Storage>,
    sensors: SubBuffer<Sensor, Storage>,
    muscles: SubBuffer<Muscle, Storage>,
    persistent_floats: SubBuffer<f32, Storage>,
    particle_constants: SubBuffer<ParticleConstants, Storage>,
    collision_grid: SubBuffer<u32, Storage>,
    indirect: Indirect,
    sampler: Sampler,
}

impl Foundations {
    pub fn face_count_per_chunk_buffer(&self)->&SubBuffer<Face, Storage>{
        &self.face_count_per_chunk_buffer
    }
    pub fn opaque_and_transparent_face_buffer(&self)->&SubBuffer<Face, Storage>{
        &self.opaque_and_transparent_face_buffer
    }
    pub fn world_size(&self) -> &WorldSize {
        &self.world_size
    }
    pub fn indirect(&self) -> &Indirect {
        &self.indirect
    }
    pub fn particles(&self) -> &SubBuffer<Particle, Storage> {
        &self.particles
    }
    pub fn bones(&self) -> &SubBuffer<Bone, Storage> {
        &self.bones
    }
    pub fn sensors(&self) -> &SubBuffer<Sensor, Storage> {
        &self.sensors
    }
    pub fn muscles(&self) -> &SubBuffer<Muscle, Storage> {
        &self.muscles
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
    pub fn block_properties(&self) -> &SubBuffer<BlockProp, Storage> {
        &self.block_properties
    }
    pub fn faces(&self) -> &SubBuffer<Face, Storage> {
        &self.faces
    }
    pub fn constants(&self) -> &SubBuffer<ParticleConstants, Storage> {
        &self.particle_constants
    }
    pub fn constraints(&self) -> &SubBuffer<Constraint, Storage> {
        &self.constraints
    }

    pub fn collision_grid(&self) -> &SubBuffer<u32, Storage> {
        &self.collision_grid
    }
}