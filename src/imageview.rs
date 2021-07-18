use ash::vk;
use crate::device::Device;
use ash::version::DeviceV1_0;
use crate::render_pass::RenderPass;
use crate::swap_chain::SwapChain;

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
}

impl Drop for Framebuffer {
    fn drop(&mut self) {
        unsafe { self.device.inner().destroy_framebuffer(self.raw, None); }
    }
}


pub struct ImageView {
    img: vk::ImageView,
    device: Device,
}

impl ImageView {
    pub fn device(&self) -> &Device {
        &self.device
    }
    pub fn new(raw: vk::Image, format: vk::Format, device: &Device) -> Result<Self, ash::vk::Result> {
        let imageview_create_info = vk::ImageViewCreateInfo::builder()
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image(raw);
        unsafe { device.inner().create_image_view(&imageview_create_info, None) }.map(|img| Self { img, device: device.clone() })
    }
    pub fn create_framebuffer(&self, render_pass: &RenderPass,
                              swapchain: &SwapChain) -> Result<Framebuffer, ash::vk::Result> {
        let attachments = [self.img];
        let framebuffer_create_info = vk::FramebufferCreateInfo::builder()
            .render_pass(render_pass.raw())
            .attachments(&attachments)
            .width(swapchain.extent().width)
            .height(swapchain.extent().height)
            .layers(1);

        unsafe {
            self.device.inner().create_framebuffer(&framebuffer_create_info, None)
        }.map(|raw| Framebuffer { raw, device: self.device.clone(), render_pass: render_pass.clone() })
    }
}

impl Drop for ImageView {
    fn drop(&mut self) {
        unsafe { self.device.inner().destroy_image_view(self.img, None); }
    }
}