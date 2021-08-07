use crate::render::descriptors::{DescriptorsBuilder, DescriptorsBuilderLocked, Descriptors};
use crate::render::command_pool::{CommandPool, CommandBuffer};
use crate::render::single_render_pass::SingleRenderPass;
use crate::pipelines::foundations::{FoundationInitializer, Foundations};
use crate::render::swap_chain::SwapchainImageIdx;
use crate::pipelines::player::Player;

pub trait RenderResources:Sized{
    type Render:Renderable;
    fn create_descriptors(&self,descriptors:&mut DescriptorsBuilder, foundations:&FoundationInitializer)->Result<(),failure::Error>;
    fn make_renderable(self, cmd_pool: &CommandPool, render_pass: &SingleRenderPass, descriptors:&DescriptorsBuilderLocked, foundations:&Foundations) -> Result<Self::Render, failure::Error>;

}
pub trait Renderable:Sized{
    fn record_cmd_buffer(&self, cmd: &mut CommandBuffer, image_idx:SwapchainImageIdx,descriptors:&Descriptors, render_pass:&SingleRenderPass, foundations:&Foundations)->Result<(),failure::Error>;
    fn record_compute_cmd_buffer(&self, cmd: &mut CommandBuffer, foundations:&Foundations)->Result<(),failure::Error>;
    fn update_uniforms(&mut self, image_idx:SwapchainImageIdx, player:&Player);
    fn recreate(&mut self, render_pass: &SingleRenderPass, ) -> Result<(), failure::Error>;
}