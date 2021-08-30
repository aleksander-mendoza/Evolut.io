use crate::render::stage_buffer::{StageBuffer, StageSubBuffer, IndirectDispatchSubBuffer, IndirectSubBuffer};
use crate::pipelines::particle::Particle;
use crate::render::command_pool::{CommandPool};


use ash::vk;


use failure::Error;


use crate::render::submitter::{Submitter, fill_submit};

use crate::render::buffer_type::{Cpu, Storage, GpuIndirect};

use crate::blocks::world_size::{CHUNK_VOLUME_IN_CELLS, CHUNK_WIDTH, CHUNK_DEPTH};
use crate::render::subbuffer::SubBuffer;
use crate::pipelines::constraint::Constraint;
use crate::render::buffer::Buffer;
use crate::pipelines::particle_constants::ParticleConstants;
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

pub struct Indirect {
    collision_detection: SubBuffer<vk::DispatchIndirectCommand, GpuIndirect>,
    solve_constraints: SubBuffer<vk::DispatchIndirectCommand, GpuIndirect>,
    update_bones: SubBuffer<vk::DispatchIndirectCommand, GpuIndirect>,
    agent_sensory_input_update: SubBuffer<vk::DispatchIndirectCommand, GpuIndirect>,
    feed_forward_net: SubBuffer<vk::DispatchIndirectCommand, GpuIndirect>,
    draw_bones: SubBuffer<vk::DrawIndirectCommand, GpuIndirect>,
    draw_blocks: SubBuffer<vk::DrawIndirectCommand, GpuIndirect>,
    draw_particles: SubBuffer<vk::DrawIndirectCommand, GpuIndirect>,
    super_indirect_buffer: SubBuffer<u8, GpuIndirect>,
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

    world: Submitter<StageSubBuffer<Block, Cpu, Storage>>,
    faces: Submitter<StageSubBuffer<Face, Cpu, Storage>>,
    face_count_per_chunk_buffer: SubBuffer<Face, Storage>,
    opaque_and_transparent_face_buffer: SubBuffer<Face, Storage>,
    particles: Submitter<StageSubBuffer<Particle, Cpu, Storage>>,
    // collision_grid: Submitter<SubBuffer<u32, Storage>>,
    bones: Submitter<StageSubBuffer<Bone, Cpu, Storage>>,
    // sensors: Submitter<StageSubBuffer<Sensor, Cpu, Storage>>,
    // muscles: Submitter<StageSubBuffer<Muscle, Cpu, Storage>>,
    persistent_floats: SubBuffer<f32, Storage>,
    // zombie_brains: ZombieNeat<f32>,
    neural_net_layers: Submitter<StageSubBuffer<NeuralNetLayer, Cpu, Storage>>,
    // constraints: Submitter<StageSubBuffer<Constraint, Cpu, Storage>>,
    block_properties: Submitter<StageSubBuffer<BlockProp, Cpu, Storage>>,
    particle_constants: Submitter<StageSubBuffer<ParticleConstants, Cpu, Storage>>,
    indirect_dispatch: Submitter<IndirectDispatchSubBuffer>,
    indirect_draw: Submitter<IndirectSubBuffer>,
    indirect: Indirect,
    sampler: Sampler,
    world_size: WorldSize,
}

struct FoundationsCapacity {
    world_size: WorldSize,
    particles: u64,
    bones: u64,
    faces: u64,
    max_constraints: u64,
    // solid_particles: u64,
    // phantom_particles: u64,
    sensors: u64,
    persistent_floats: u64,
    neural_net_layers: u64,
    muscles: u64,
}

impl FoundationsCapacity {
    pub fn new() -> Self {
        let world_size = WorldSize::new(2,2);
        let particles = 256u64;
        let bones = 256u64;
        let faces = 16 * 1024u64 * world_size.total_chunks() as u64;
        let max_constraints = 2048u64;
        let sensors = 1024u64;
        let persistent_floats = 64*1024u64; // used as backing memory for vectors, matrices and
        // tensors that make up various neural networks. Especially, the outputs of recursive neural networks
        // often need to be persistent, because those outputs are later fed as inputs to the same neural net.
        let neural_net_layers = 128u64;
        let muscles = 1024u64;
        // debug_assert!(solid_particles + phantom_particles <= particles,"{} + {} <= {}",solid_particles, phantom_particles, particles);
        Self {
            world_size,
            particles,
            bones,
            faces,
            max_constraints,
            sensors,
            persistent_floats,
            neural_net_layers,
            muscles,
        }
    }
}

const SUBSTRATE_IN_DIM:usize = 2;
const SUBSTRATE_OUT_DIM:usize = 2;
const SUBSTRATE_WEIGHT_DIM:usize = 1;

struct ZombieNeat<X:Num>{
    substrate_in_positions:Vec<[X; SUBSTRATE_IN_DIM]>,
    substrate_out_positions:Vec<[X; SUBSTRATE_OUT_DIM]>,
    neat:Neat<X>,
    population:Vec<ZombieBrain<X>>,
}

impl <X:Num> ZombieNeat<X>{
    fn into_brain(zombie:Zombie,cppn:CPPN<X>,
                          cmd_pool:&CommandPool,
                          substrate_in_positions:&Vec<[X; SUBSTRATE_IN_DIM]>,
                          substrate_out_positions:&Vec<[X; SUBSTRATE_OUT_DIM]>,
                          persistent_floats_buffer:&SubBuffer<X,Storage>)->Result<ZombieBrain<X>,vk::Result>{

        let net = cppn.build_feed_forward_net();
        let mut weights = Vec::with_capacity(zombie.weights_len as usize);
        for i in 0..(zombie.sensors_len+zombie.recurrent_len) as usize{
            for o in 0..(zombie.muscles_constraints_len+zombie.recurrent_len) as usize{
                let mut weight = [X::zero()];
                net.run(&[substrate_in_positions[i][0],substrate_in_positions[i][1],
                    substrate_out_positions[o][0],substrate_out_positions[o][1]],&mut weight );
                weights.push(weight[0]);
            }
        }

        let buff = StageBuffer::wrap( cmd_pool, weights.as_slice(), persistent_floats_buffer.sub_elem(zombie.weights_offset as u64, zombie.weights_len as u64))?;
        Ok(ZombieBrain{zombie,cppn,buff})
    }
    fn new(zombies:Vec<Zombie>,cmd_pool:&CommandPool, persistent_floats:&SubBuffer<X,Storage>)->Result<Self,vk::Result>{
        let substrate_in_positions = (0.. zombies[0].sensors_len+zombies[0].recurrent_len).map(|_|{
            let mut arr = [X::zero(); SUBSTRATE_IN_DIM];
            for a in &mut arr{*a = X::random()}
            arr
        }).collect::<Vec<[X; SUBSTRATE_IN_DIM]>>();
        let substrate_out_positions = (0.. zombies[0].muscles_constraints_len+zombies[0].recurrent_len).map(|_|{
            let mut arr = [X::zero(); SUBSTRATE_IN_DIM];
            for a in &mut arr{*a = X::random()}
            arr
        }).collect::<Vec<[X; SUBSTRATE_OUT_DIM]>>();
        let mut neat = Neat::new(vec![X::ACT_FN_ABS, X::ACT_FN_SIN, X::ACT_FN_CONST_1, X::ACT_FN_TANH, X::ACT_FN_GAUSSIAN, X::ACT_FN_SQUARE], SUBSTRATE_IN_DIM + SUBSTRATE_OUT_DIM, SUBSTRATE_WEIGHT_DIM);
        let mut cppns = neat.new_cppns(zombies.len());
        for _ in 0..32{
            for cppn in &mut cppns{
                neat.mutate(cppn,0.1,0.1,0.1,0.1,0.1,0.1);
            }
        }
        // println!("{}",cppns[0]);
        let zombies:Result<Vec<ZombieBrain<X>>,vk::Result> = zombies.into_iter().zip(cppns.into_iter()).map(|(z,c)|Self::into_brain(z,c,
                                                                                             cmd_pool,
                                                                                             &substrate_in_positions,
                                                                                             &substrate_out_positions,
                                                                                             persistent_floats
        )).collect();
        let population = zombies?;
        // println!("{:?}",(&population[0]).buff.as_slice());
        Ok(Self{
            substrate_in_positions,
            substrate_out_positions,
            neat,
            population
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

struct ZombieBrain<X:Num>{
    zombie:Zombie,
    cppn:CPPN<X>,
    buff:Submitter<StageSubBuffer<X,Cpu,Storage>>,
}

struct FoundationsData {
    world_blocks: WorldBlocks,
    world_faces: WorldFaces,
    particle_data: Vec<Particle>,
    constraints_data: Vec<Constraint>,
    bone_data: Vec<Bone>,
    sensor_data: Vec<Sensor>,
    muscle_data: Vec<Muscle>,
    block_properties_data: Vec<BlockProp>,
    neural_net_layer_data: Vec<NeuralNetLayer>,
    persistent_floats_len: u32,
}

impl FoundationsData {
    pub fn new(cap: &FoundationsCapacity) -> Self {
        Self {
            world_blocks: WorldBlocks::new(cap.world_size.clone()),
            world_faces: WorldFaces::with_capacity(cap.world_size.clone(), cap.faces as usize),
            // particles_data: std::iter::repeat_with(Particle::random).take((cap.solid_particles+cap.phantom_particles) as usize).collect(),
            particle_data: vec![],
            constraints_data: vec![],
            bone_data: vec![],
            sensor_data: vec![],
            muscle_data: vec![],
            neural_net_layer_data: vec![],
            block_properties_data: BLOCKS.iter().map(|p| p.prop).collect(),
            persistent_floats_len: 0,
        }
    }
    fn setup_world_blocks(&mut self) {
        const SEA_LEVEL:usize = 128;
        let size_with_margins = WorldSize::new(self.world_blocks.size().width()+2,self.world_blocks.size().depth()+2);
        let chunk_heights:Vec<f32> = (0..size_with_margins.total_chunks()).map(|_|100. + f32::random()*60.).collect();
        for x in 0..self.world_blocks.size().world_width(){
            for z in 0..self.world_blocks.size().world_depth(){
                let pos_with_margins_x = x + CHUNK_WIDTH;
                let pos_with_margins_z = z + CHUNK_DEPTH;
                let radius_x =CHUNK_WIDTH/2-1;
                let radius_z = CHUNK_DEPTH/2-1;
                let fraction_x = ((pos_with_margins_x as f32 + 0.5 - CHUNK_WIDTH as f32/2f32) / (CHUNK_WIDTH as f32)).fract();
                let fraction_z = ((pos_with_margins_z as f32 + 0.5 - CHUNK_DEPTH as f32/2f32) / (CHUNK_DEPTH as f32)).fract();
                let neighbour_height_right_top = chunk_heights[size_with_margins.block_pos_into_chunk_idx(pos_with_margins_x+radius_x,pos_with_margins_z+radius_z)];
                let neighbour_height_right_bottom = chunk_heights[size_with_margins.block_pos_into_chunk_idx(pos_with_margins_x+radius_x,pos_with_margins_z-radius_z)];
                let neighbour_height_left_top = chunk_heights[size_with_margins.block_pos_into_chunk_idx(pos_with_margins_x-radius_x,pos_with_margins_z+radius_z)];
                let neighbour_height_left_bottom = chunk_heights[size_with_margins.block_pos_into_chunk_idx(pos_with_margins_x-radius_x,pos_with_margins_z-radius_z)];
                let height_left = fraction_z.smoothstep_between(neighbour_height_left_bottom, neighbour_height_left_top);
                let height_right = fraction_z.smoothstep_between(neighbour_height_right_bottom, neighbour_height_right_top);
                let height = fraction_x.smoothstep_between(height_left, height_right) as usize;
                self.world_blocks.fill_column_to(x,1,z,height  - 4, STONE);
                self.world_blocks.fill_column_to(x,height - 4,z,height, DIRT);
                if height < SEA_LEVEL{
                    self.world_blocks.fill_column_to(x,height ,z,SEA_LEVEL+1 ,WATER);
                }else{
                    self.world_blocks.set_block(x,height ,z, GRASS);
                }

            }
        }


        self.world_blocks.fill_level(0, 1, BEDROCK);


    }
    fn setup_world_faces(&mut self) {
        let Self { world_blocks, world_faces, .. } = self;
        world_faces.generate_faces(world_blocks);
    }

    fn compute_constants(&self, cap: &FoundationsCapacity) -> ParticleConstants {
        let FoundationsCapacity { world_size, .. } = cap;
        let Self { sensor_data, constraints_data, particle_data: phantom_particle_data, bone_data, .. } = self;
        ParticleConstants {
            predefined_constraints: constraints_data.len() as i32,
            collision_constraints: 0,
            particles: phantom_particle_data.len() as i32,
            dummy: 0,
            chunks_x: world_size.width() as i32,
            chunks_z: world_size.depth() as i32,
            bones: bone_data.len() as i32,
            world_width: world_size.world_width() as i32,
            world_depth: world_size.world_depth() as i32,
            world_area: world_size.world_area() as i32,
            total_chunks: world_size.total_chunks() as i32,
            sensors: sensor_data.len() as u32,
        }
    }

}

fn append_owned<X>(v: &mut Vec<X>, mut v2: Vec<X>) {
    v.append(&mut v2);
}

impl FoundationInitializer {

    pub fn face_count_per_chunk_buffer(&self) -> &SubBuffer<Face, Storage> {
        &self.face_count_per_chunk_buffer
    }
    pub fn opaque_and_transparent_face_buffer(&self) -> &SubBuffer<Face, Storage> {
        &self.opaque_and_transparent_face_buffer
    }
    pub fn particle_constants(&self) -> &StageSubBuffer<ParticleConstants, Cpu, Storage> {
        &self.particle_constants
    }
    // pub fn sensors(&self) -> &StageSubBuffer<Sensor, Cpu, Storage> {
    //     &self.sensors
    // }
    pub fn persistent_floats(&self) -> &SubBuffer<f32, Storage> {
        &self.persistent_floats
    }
    // pub fn muscles(&self) -> &StageSubBuffer<Muscle, Cpu, Storage> {
    //     &self.muscles
    // }
    //
    pub fn particles(&self) -> &StageSubBuffer<Particle, Cpu, Storage> {
        &self.particles
    }
    // pub fn constraints(&self) -> &StageSubBuffer<Constraint, Cpu, Storage> {
    //     &self.constraints
    // }
    // pub fn collision_grid(&self) -> &SubBuffer<u32, Storage> {
    //     &self.collision_grid
    // }
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
    pub fn faces(&self) -> &StageSubBuffer<Face, Cpu, Storage> {
        &self.faces
    }

    pub fn block_properties(&self) -> &StageSubBuffer<BlockProp, Cpu, Storage> {
        &self.block_properties
    }
    pub fn new(cmd_pool: &CommandPool) -> Result<Self, failure::Error> {
        let cap = FoundationsCapacity::new();
        let mut data = FoundationsData::new(&cap);
        data.setup_world_blocks();
        data.setup_world_faces();
        data.particle_data.extend((0..64).map(|_| Particle::random()));
        data.bone_data.push(Bone::new(glm::vec3(5.,128.,5.),glm::vec3(0.5,1.0,0.5),glm::vec3(0.1,1.0,0.1),glm::vec3(0.1,0.1,0.9), 1.0));
        data.neural_net_layer_data.append(&mut vec![
            NeuralNetLayer::new_input_recurrent(0,0,0,0),
            NeuralNetLayer::new_hidden(0,0,0,0,0,None,Overwrite),
            NeuralNetLayer::new_output(0,0,0,0,0,0)
        ]);
        // let mut zombies = vec![];
        // for r in 0..5 {
        //     for c in 0..1 {
        //         zombies.push(data.add_zombie(glm::vec3(2. + r as f32 * 3., 5., 2. + c as f32 * 3.), &cap));
        //     }
        // }


        let constants = data.compute_constants(&cap);
        println!("{:?}", constants);
        assert!(cap.persistent_floats >= data.persistent_floats_len as u64,"{} < {}",cap.persistent_floats, data.persistent_floats_len);
        let particles_in_bytes = std::mem::size_of::<Particle>() as u64 * cap.particles;
        let faces_in_bytes = std::mem::size_of::<Face>() as u64 * cap.faces;
        // let grid_in_bytes = std::mem::size_of::<u32>() as u64 * cap.grid_size;
        let block_properties_in_bytes = std::mem::size_of_val(data.block_properties_data.as_slice()) as u64;
        let world_in_bytes = (std::mem::size_of::<Block>() * cap.world_size.world_volume()) as u64;
        // let constraints_in_bytes = std::mem::size_of::<Constraint>() as u64 * cap.max_constraints;
        let bones_in_bytes = std::mem::size_of::<Bone>() as u64 * cap.bones;
        let constants_in_bytes = std::mem::size_of_val(&constants) as u64;
        // let sensors_in_bytes = std::mem::size_of::<Sensor>() as u64 * cap.sensors;
        let persistent_floats_in_bytes = std::mem::size_of::<f32>() as u64 * cap.persistent_floats;
        let neural_net_layers_in_bytes = std::mem::size_of::<NeuralNetLayer>() as u64 * cap.neural_net_layers;
        // let muscles_in_bytes = std::mem::size_of::<Muscle>() as u64 * cap.muscles;

        let super_buffer: SubBuffer<u8, Storage> = SubBuffer::with_capacity(cmd_pool.device(),
                                                                            particles_in_bytes +
                                                                                faces_in_bytes +
                                                                                // grid_in_bytes +
                                                                                block_properties_in_bytes +
                                                                                world_in_bytes +
                                                                                // constraints_in_bytes +
                                                                                bones_in_bytes +
                                                                                constants_in_bytes +
                                                                                // sensors_in_bytes +
                                                                                persistent_floats_in_bytes +
                                                                                neural_net_layers_in_bytes
                                                                                // + muscles_in_bytes
        )?;
        let offset = 0;
        let particle_buffer = super_buffer.sub(offset..offset + particles_in_bytes).reinterpret_into::<Particle>();
        let offset = offset + particles_in_bytes;
        assert_eq!(offset % 16, 0); // check correct GLSL alignment
        let face_buffer = super_buffer.sub(offset..offset + faces_in_bytes).reinterpret_into::<Face>();
        let offset = offset + faces_in_bytes;
        assert_eq!(offset % 16, 0);
        // let grid_buffer = super_buffer.sub(offset..offset + grid_in_bytes).reinterpret_into::<u32>();
        // let offset = offset + grid_in_bytes;
        // assert_eq!(offset % 16, 0);
        let block_properties_buffer = super_buffer.sub(offset..offset + block_properties_in_bytes).reinterpret_into::<BlockProp>();
        let offset = offset + block_properties_in_bytes;
        assert_eq!(offset % 16, 0);
        let world_buffer = super_buffer.sub(offset..offset + world_in_bytes).reinterpret_into::<Block>();
        let offset = offset + world_in_bytes;
        assert_eq!(offset % 16, 0);
        // let constraint_buffer = super_buffer.sub(offset..offset + constraints_in_bytes).reinterpret_into::<Constraint>();
        // let offset = offset + constraints_in_bytes;
        // assert_eq!(offset % 16, 0);
        let bones_buffer = super_buffer.sub(offset..offset + bones_in_bytes).reinterpret_into::<Bone>();
        let offset = offset + bones_in_bytes;
        assert_eq!(offset % 16, 0);
        let constants_buffer = super_buffer.sub(offset..offset + constants_in_bytes).reinterpret_into::<ParticleConstants>();
        let offset = offset + constants_in_bytes;
        assert_eq!(offset % 16, 0);
        // let sensors_buffer = super_buffer.sub(offset..offset + sensors_in_bytes).reinterpret_into::<Sensor>();
        // let offset = offset + sensors_in_bytes;
        // assert_eq!(offset % 16, 0);
        let persistent_floats_buffer = super_buffer.sub(offset..offset + persistent_floats_in_bytes).reinterpret_into::<f32>();
        let offset = offset + persistent_floats_in_bytes;
        assert_eq!(offset % 16, 0);
        let neural_net_layers_buffer = super_buffer.sub(offset..offset + neural_net_layers_in_bytes).reinterpret_into::<NeuralNetLayer>();
        let offset = offset + neural_net_layers_in_bytes;
        assert_eq!(offset % 16, 0);
        // let muscles_buffer = super_buffer.sub(offset..offset + muscles_in_bytes).reinterpret_into::<Muscle>();
        // let offset = offset + muscles_in_bytes;
        // assert_eq!(offset % 16, 0);

        let particle_constants = StageBuffer::wrap(cmd_pool, &[constants], constants_buffer)?;

        let particles = StageBuffer::wrap(cmd_pool, &data.particle_data, particle_buffer)?;

        let face_count_per_chunk_buffer = face_buffer.sub(..std::mem::size_of::<Face>() as u64 * cap.world_size.total_chunks() as u64 * 2);
        let opaque_and_transparent_face_buffer = face_buffer.sub(std::mem::size_of::<Face>() as u64 * cap.world_size.total_chunks() as u64 * 2..);
        let faces = StageBuffer::wrap(cmd_pool, data.world_faces.as_slice(), face_buffer)?;

        // let mut collision_grid = Submitter::new(grid_buffer, cmd_pool)?;
        // fill_submit(&mut collision_grid, u32::MAX)?;

        let block_properties = StageBuffer::wrap(cmd_pool, &data.block_properties_data, block_properties_buffer)?;

        let bones = StageBuffer::wrap(cmd_pool, &data.bone_data, bones_buffer)?;

        // let sensors = StageBuffer::wrap(cmd_pool, &data.sensor_data, sensors_buffer)?;

        // let zombie_brains = ZombieNeat::new(zombies,cmd_pool,&persistent_floats_buffer)?;

        let persistent_floats = persistent_floats_buffer;

        let world = StageBuffer::wrap(cmd_pool, data.world_blocks.as_slice(), world_buffer)?;

        // let constraints = StageBuffer::wrap(cmd_pool, &data.constraints_data, constraint_buffer)?;

        let neural_net_layers = StageBuffer::wrap(cmd_pool, &data.neural_net_layer_data, neural_net_layers_buffer)?;

        // let muscles = StageBuffer::wrap(cmd_pool, &data.muscle_data, muscles_buffer)?;

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
            dispatch_indirect(data.particle_data.len()),// collision_detection.comp
            dispatch_indirect(0), // solve_constraints.comp
            dispatch_indirect(data.bone_data.len()), // update_bones.comp
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
            draw_indirect(data.particle_data.len() as u32, 1),// particles.vert

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
            world_size: cap.world_size.clone(),
            sampler,
            particles,
            // constraints,
            // sensors,
            // zombie_brains,
            persistent_floats,
            neural_net_layers,
            // collision_grid,
            particle_constants,
            indirect_dispatch,
            indirect_draw,
            // muscles,
            indirect,
            bones,
        })
    }
    pub fn build(self) -> Result<Foundations, Error> {
        let Self {
            face_count_per_chunk_buffer,
            opaque_and_transparent_face_buffer,
            block_properties,
            // zombie_brains,
            indirect_dispatch,
            indirect_draw,
            world_size,
            faces,
            world,
            bones,
            // sensors,
            persistent_floats,
            particles,
            // collision_grid,
            // constraints,
            particle_constants,
            indirect,
            neural_net_layers,
            // muscles,
            sampler,
        } = self;
        let particles = particles.take()?.take_gpu();
        // let collision_grid = collision_grid.take()?;
        let block_properties = block_properties.take()?.take_gpu();
        // let constraints = constraints.take()?.take_gpu();
        let bones = bones.take()?.take_gpu();
        // let sensors = sensors.take()?.take_gpu();
        // let muscles = muscles.take()?.take_gpu();
        let neural_net_layers = neural_net_layers.take()?.take_gpu();
        let particle_constants = particle_constants.take()?.take_gpu();
        let _ = indirect_dispatch.take()?.take_gpu();
        let _ = indirect_draw.take()?.take_gpu();
        let world = world.take()?.take_gpu(); //wait for completion and then dispose of the staging buffer
        let faces = faces.take()?.take_gpu();
        Ok(Foundations {
            // muscles,
            face_count_per_chunk_buffer,
            opaque_and_transparent_face_buffer,
            block_properties,
            faces,
            // sensors,
            persistent_floats,
            world,
            world_size,
            bones,
            // collision_grid,
            neural_net_layers,
            particle_constants,
            // constraints,
            particles,
            indirect,
            sampler,
            // zombie_brains,
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
    // constraints: SubBuffer<Constraint, Storage>,
    bones: SubBuffer<Bone, Storage>,
    neural_net_layers: SubBuffer<NeuralNetLayer, Storage>,
    // sensors: SubBuffer<Sensor, Storage>,
    // muscles: SubBuffer<Muscle, Storage>,
    persistent_floats: SubBuffer<f32, Storage>,
    particle_constants: SubBuffer<ParticleConstants, Storage>,
    // collision_grid: SubBuffer<u32, Storage>,
    indirect: Indirect,
    sampler: Sampler,
    // zombie_brains:ZombieNeat<f32>,
}

impl Foundations {
    pub fn face_count_per_chunk_buffer(&self) -> &SubBuffer<Face, Storage> {
        &self.face_count_per_chunk_buffer
    }
    pub fn opaque_and_transparent_face_buffer(&self) -> &SubBuffer<Face, Storage> {
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
    // pub fn sensors(&self) -> &SubBuffer<Sensor, Storage> {
    //     &self.sensors
    // }
    // pub fn muscles(&self) -> &SubBuffer<Muscle, Storage> {
    //     &self.muscles
    // }
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
    // pub fn constraints(&self) -> &SubBuffer<Constraint, Storage> {
    //     &self.constraints
    // }

    // pub fn collision_grid(&self) -> &SubBuffer<u32, Storage> {
    //     &self.collision_grid
    // }
}