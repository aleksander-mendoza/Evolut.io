// use crate::neat::cppn::CPPN;
// use crate::render::command_pool::CommandPool;
// use crate::neat::num::Num;
// use crate::render::subbuffer::SubBuffer;
// use crate::render::buffer_type::{Storage, Cpu};
// use ash::vk;
// use crate::neat::neat::Neat;
// use crate::render::submitter::Submitter;
// use crate::render::stage_buffer::StageSubBuffer;
// use crate::neat::entity::Entity;
// use crate::render::stage_buffer::StageBuffer;
// const SUBSTRATE_IN_DIM: usize = 2;
// const SUBSTRATE_OUT_DIM: usize = 2;
// const SUBSTRATE_WEIGHT_DIM: usize = 1;
//
// pub struct EntityNeat<X: Num> {
//     substrate_in_positions: Vec<[X; SUBSTRATE_IN_DIM]>,
//     substrate_out_positions: Vec<[X; SUBSTRATE_OUT_DIM]>,
//     neat: Neat<X>,
//     population: Vec<EntityBrain<X>>,
// }
//
// impl<X: Num> EntityNeat<X> {
//     fn into_brain(entity: Entity, cppn: CPPN<X>,
//                   cmd_pool: &CommandPool,
//                   substrate_in_positions: &Vec<[X; SUBSTRATE_IN_DIM]>,
//                   substrate_out_positions: &Vec<[X; SUBSTRATE_OUT_DIM]>,
//                   persistent_floats_buffer: &SubBuffer<X, Storage>) -> Result<EntityBrain<X>, vk::Result> {
//         let net = cppn.build_feed_forward_net();
//         let mut weights = Vec::with_capacity(entity.weights_len as usize);
//         for i in 0..(entity.sensors_len + entity.recurrent_len) as usize {
//             for o in 0..(entity.muscles_constraints_len + entity.recurrent_len) as usize {
//                 let mut weight = [X::zero()];
//                 net.run(&[substrate_in_positions[i][0], substrate_in_positions[i][1],
//                     substrate_out_positions[o][0], substrate_out_positions[o][1]], &mut weight);
//                 weights.push(weight[0]);
//             }
//         }
//
//         let buff = StageBuffer::wrap(cmd_pool, weights.as_slice(), persistent_floats_buffer.sub_elem(entity.weights_offset as u64, entity.weights_len as u64))?;
//         Ok(EntityBrain { entity, cppn, buff })
//     }
//     fn new(entitys: Vec<Entity>, cmd_pool: &CommandPool, persistent_floats: &SubBuffer<X, Storage>) -> Result<Self, vk::Result> {
//         let substrate_in_positions = (0..entitys[0].sensors_len + entitys[0].recurrent_len).map(|_| {
//             let mut arr = [X::zero(); SUBSTRATE_IN_DIM];
//             for a in &mut arr { *a = X::random() }
//             arr
//         }).collect::<Vec<[X; SUBSTRATE_IN_DIM]>>();
//         let substrate_out_positions = (0..entitys[0].muscles_constraints_len + entitys[0].recurrent_len).map(|_| {
//             let mut arr = [X::zero(); SUBSTRATE_IN_DIM];
//             for a in &mut arr { *a = X::random() }
//             arr
//         }).collect::<Vec<[X; SUBSTRATE_OUT_DIM]>>();
//         let mut neat = Neat::new(vec![X::ACT_FN_ABS, X::ACT_FN_SIN, X::ACT_FN_CONST_1, X::ACT_FN_TANH, X::ACT_FN_GAUSSIAN, X::ACT_FN_SQUARE], SUBSTRATE_IN_DIM + SUBSTRATE_OUT_DIM, SUBSTRATE_WEIGHT_DIM);
//         let mut cppns = neat.new_cppns(entitys.len());
//         for _ in 0..32 {
//             for cppn in &mut cppns {
//                 neat.mutate(cppn, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1);
//             }
//         }
//         // println!("{}",cppns[0]);
//         let entitys: Result<Vec<EntityBrain<X>>, vk::Result> = entitys.into_iter().zip(cppns.into_iter()).map(|(z, c)| Self::into_brain(z, c,
//                                                                                                                                         cmd_pool,
//                                                                                                                                         &substrate_in_positions,
//                                                                                                                                         &substrate_out_positions,
//                                                                                                                                         persistent_floats,
//         )).collect();
//         let population = entitys?;
//         // println!("{:?}",(&population[0]).buff.as_slice());
//         Ok(Self {
//             substrate_in_positions,
//             substrate_out_positions,
//             neat,
//             population,
//         })
//     }
// }
//
//
// struct EntityBrain<X: Num> {
//     entity: Entity,
//     cppn: CPPN<X>,
//     buff: Submitter<StageSubBuffer<X, Cpu, Storage>>,
// }