use ash::vk;
use crate::device::Device;
use crate::render_pass::RenderPass;
use ash::version::DeviceV1_0;
use crate::swap_chain::SwapChain;
use crate::imageview::{ImageView, Depth, Color};
use crate::texture::Dim2D;

pub struct Framebuffer {
    raw: vk::Framebuffer,
    device: Device,
    render_pass: RenderPass,
}

impl Framebuffer {
    pub fn raw(&self) -> vk::Framebuffer {
        self.raw
    }
    pub fn device(&self) -> &Device {
        &self.device
    }
    pub fn new_color_and_depth(render_pass: &RenderPass, swapchain: &SwapChain, color:&ImageView<Color>, depth:&ImageView<Depth>) -> Result<Framebuffer, ash::vk::Result> {
        Self::new(render_pass,swapchain,&[color.raw(),depth.raw()])
    }

    pub fn new(render_pass: &RenderPass, swapchain: &SwapChain, attachments:&[vk::ImageView]) -> Result<Framebuffer, ash::vk::Result> {
        let framebuffer_create_info = vk::FramebufferCreateInfo::builder()
            .render_pass(render_pass.raw())
            .attachments(attachments)
            .width(swapchain.extent().width)
            .height(swapchain.extent().height)
            .layers(1);

        unsafe {
            render_pass.device().inner().create_framebuffer(&framebuffer_create_info, None)
        }.map(|raw| Framebuffer { raw, device: render_pass.device().clone(), render_pass: render_pass.clone() })
    }
}

impl Drop for Framebuffer {
    fn drop(&mut self) {
        unsafe { self.device.inner().destroy_framebuffer(self.raw, None); }
    }
}