use crate::render::descriptors::{DescriptorsBuilder, DescriptorsBuilderLocked, Descriptors};
use crate::render::command_pool::{CommandPool, CommandBuffer};
use crate::render::single_render_pass::SingleRenderPass;
use crate::pipelines::foundations::{FoundationInitializer, Foundations};
use crate::render::swap_chain::SwapchainImageIdx;
use crate::pipelines::player::Player;

pub trait ComputeResources:Sized{
    type Compute:Computable;
    fn make_computable(self, cmd_pool: &CommandPool, foundations:&Foundations) -> Result<Self::Compute, failure::Error>;

}
pub trait Computable:Sized{
    fn record_compute_cmd_buffer(&self, cmd: &mut CommandBuffer, foundations:&Foundations)->Result<(),failure::Error>;
    fn update_uniforms(&mut self, player:&Player);
}