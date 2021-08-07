// use crate::pipelines::computable::{Computable, ComputeResources};
// use crate::pipelines::foundations::Foundations;
// use std::sync::Arc;
// use crate::render::device::Device;
// use crate::pipelines::renderable::RenderResources;
// use crate::pipelines::display::Display;
// use crate::render::command_pool::CommandPool;
// use std::sync::mpsc::Receiver;
//
// pub struct ComputeThreadBuilder{
//     foundations:Arc<Foundations>,
//     device:Device
// }
//
// impl ComputeThreadBuilder{
//     pub fn new<P: RenderResources>(display:&Display<P>)->Self{
//         let foundations = display.foundations();
//         let device = display.device().clone();
//         Self{foundations,device}
//     }
//     pub fn build<C:ComputeResources,E>(self, tx:Receiver<E>, compute:impl FnOnce(&CommandPool, &Foundations)->Result<C,failure::Error>)->Result<ComputeThread<C::Compute,E>,failure::Error>{
//         let Self{ foundations, device } = self;
//         let cmd_pool = CommandPool::new(&device, true)?;
//         let compute_resources = compute(&cmd_pool,&foundations)?;
//         let compute = compute_resources.make_computable(&cmd_pool, &foundations)?;
//         Ok(ComputeThread{ tx, foundations, cmd_pool, device, compute })
//     }
//     pub fn device(&self)->&Device{
//         &self.device
//     }
//     pub fn foundations(&self)->&Foundations{
//         &self.foundations
//     }
// }
//
// pub struct ComputeThread<C:Computable,E>{
//     foundations:Arc<Foundations>,
//     cmd_pool: CommandPool,
//     device:Device,
//     compute:C,
//     tx:Receiver<E>,
// }
//
// impl <C:Computable,E> ComputeThread<C,E>{
//     pub fn receiver(&self) -> &Receiver<E>{
//         &self.tx
//     }
//     pub fn cmd_pool(&self) -> &CommandPool{
//         &self.cmd_pool
//     }
//     pub fn compute(&self) -> &C{
//         &self.compute
//     }
//     pub fn compute_mut(&mut self) -> &mut C{
//         &mut self.compute
//     }
//     pub fn device(&self)->&Device{
//         &self.device
//     }
//     pub fn foundations(&self)->&Foundations{
//         &self.foundations
//     }
//
// }
//
//
//
// impl ComputeThread<Physics,PlayerEvent>{
//     pub fn main(self) {
//         if let Err(err) = self.run(){
//             eprintln!("{:?}",err);
//         }
//     }
//     fn run(self)->Result<(),failure::Error>{
//         let mut cmd_buff = self.cmd_pool().create_command_buffer()?;
//         let fence= Fence::new(self.device(),false)?;
//         cmd_buff.begin(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE)?;
//         self.compute().record_compute_cmd_buffer(&mut cmd_buff,self.foundations())?;
//         cmd_buff.end()?;
//         let mut timer = PhysicsTimer::new(60);
//         loop{
//             // let next_input:PlayerEvent = self.receiver().try_recv()?;
//
//             cmd_buff.submit(&[],&[],Some(&fence))?;
//             fence.wait(None)?;
//             fence.reset()?;
//             timer.update();
//         }
//     }
// }