use crate::render::render_pass::{RenderPassBuilder, RenderPass};
use ash::vk;
use crate::render::swap_chain::{SwapChain, SwapchainImageIdx};
use crate::render::texture::{TextureView, Dim2D};
use crate::render::imageview::Depth;
use std::ops::{Deref, DerefMut};
use crate::render::framebuffer::Framebuffer;
use crate::render::instance::Instance;
use crate::render::device::Device;
use crate::render::surface::Surface;

pub struct SingleRenderPass{
    framebuffers: Vec<Framebuffer>,
    depth_attachment:TextureView<Dim2D,Depth>,
    render_pass:RenderPass,
    swapchain:SwapChain,
}

impl Deref for SingleRenderPass{
    type Target = RenderPass;

    fn deref(&self) -> &Self::Target {
        &self.render_pass
    }
}

impl DerefMut for SingleRenderPass{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.render_pass
    }
}

impl SingleRenderPass{
    pub fn framebuffers_len(&self)->usize{
        self.framebuffers.len()
    }
    pub fn swapchain(&self) -> &SwapChain{
        &self.swapchain
    }
    pub fn render_pass(&self) -> &RenderPass{
        &self.render_pass
    }
    pub fn depth_attachment(&self) -> &TextureView<Dim2D,Depth>{
        &self.depth_attachment
    }
    pub fn framebuffer(&self, image_idx:SwapchainImageIdx) -> &Framebuffer{
        &self.framebuffers[image_idx.get_usize()]
    }
    pub fn new(swapchain:SwapChain)->Result<Self,failure::Error>{
        let depth_attachment = TextureView::depth_buffer_for(&swapchain)?;
        let render_pass = RenderPassBuilder::new()
            .color_attachment(swapchain.format())
            .depth_attachment(&depth_attachment)
            .graphics_subpass_with_depth([], [vk::AttachmentReference::builder()
                .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .attachment(0)
                .build()], vk::AttachmentReference::builder()
                                             .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                                             .attachment(1)
                                             .build())
            .dependency(vk::SubpassDependency {
                src_subpass: vk::SUBPASS_EXTERNAL,
                dst_subpass: 0,
                src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                src_access_mask: vk::AccessFlags::empty(),
                dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                dependency_flags: vk::DependencyFlags::empty(),
            })
            .build(swapchain.device())?;
        let framebuffers = Self::create_framebuffers(&swapchain,&render_pass,&depth_attachment)?;
        Ok(Self{render_pass,depth_attachment, swapchain, framebuffers})
    }

    pub fn new_swapchain_and_render_pass(instance: &Instance, device: &Device, surface: &Surface, old:Option<&SwapChain>)->Result<Self,failure::Error>{
        instance.create_swapchain(device, surface, old).and_then(Self::new)
    }

    fn create_framebuffers(swapchain:&SwapChain, render_pass:&RenderPass, depth_attachment:&TextureView<Dim2D,Depth>) -> Result<Vec<Framebuffer>,failure::Error>{
        swapchain.create_image_views()?.into_iter().map(|v| Framebuffer::new(render_pass, swapchain, v, depth_attachment.imageview())).collect()
    }
}