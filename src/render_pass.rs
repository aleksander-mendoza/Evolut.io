use ash::vk;
use crate::device::Device;
use ash::version::DeviceV1_0;
use std::rc::Rc;
use crate::texture::{TextureView, Dim2D};
use crate::imageview::Depth;

struct RenderPassInner {
    raw: vk::RenderPass,
    device: Device,
}

#[derive(Clone)]
pub struct RenderPass {
    inner: Rc<RenderPassInner>,
}

impl RenderPass {
    pub fn device(&self) -> &Device {
        &self.inner.device
    }
    pub fn raw(&self) -> vk::RenderPass {
        self.inner.raw
    }
}

impl Drop for RenderPassInner {
    fn drop(&mut self) {
        unsafe { self.device.inner().destroy_render_pass(self.raw, None); }
    }
}

pub struct RenderPassBuilder {
    attachments: Vec<vk::AttachmentDescription>,
    subpasses: Vec<(vk::PipelineBindPoint, Vec<vk::AttachmentReference>, Vec<vk::AttachmentReference>, Option<vk::AttachmentReference>)>,
    dependencies: Vec<vk::SubpassDependency>
}


impl RenderPassBuilder {
    pub fn color_attachment(self, surface_format: vk::Format) -> Self {
        let color_attachment = vk::AttachmentDescription {
            flags: vk::AttachmentDescriptionFlags::empty(),
            format: surface_format,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
        };
        self.attachment(color_attachment)
    }

    pub fn depth_attachment(self, depth_texture: &TextureView<Dim2D,Depth>) -> Self {
        let depth_attachment = vk::AttachmentDescription {
            flags: vk::AttachmentDescriptionFlags::empty(),
            format: depth_texture.format(),
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::DONT_CARE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };
        self.attachment(depth_attachment)
    }

    pub fn attachment(mut self, att: vk::AttachmentDescription) -> Self {
        self.attachments.push(att);
        self
    }
    pub fn graphics_subpass_with_depth(mut self, inputs: impl Into<Vec<vk::AttachmentReference>>, colors: impl Into<Vec<vk::AttachmentReference>>, depth: vk::AttachmentReference) -> Self {
        self.subpasses.push((vk::PipelineBindPoint::GRAPHICS, inputs.into(), colors.into(), Some(depth)));
        self
    }
    pub fn graphics_subpass(mut self, inputs: impl Into<Vec<vk::AttachmentReference>>, colors: impl Into<Vec<vk::AttachmentReference>>) -> Self {
        self.subpass(vk::PipelineBindPoint::GRAPHICS, inputs, colors)
    }
    pub fn compute_subpass(mut self, inputs: impl Into<Vec<vk::AttachmentReference>>, colors: impl Into<Vec<vk::AttachmentReference>>) -> Self {
        self.subpass(vk::PipelineBindPoint::COMPUTE, inputs, colors)
    }
    pub fn subpass(mut self, bind_point: vk::PipelineBindPoint, inputs: impl Into<Vec<vk::AttachmentReference>>, colors: impl Into<Vec<vk::AttachmentReference>>) -> Self {
        self.subpasses.push((bind_point, inputs.into(), colors.into(), None));
        self
    }
    pub fn dependency(mut self, dep:vk::SubpassDependency)->Self{
        self.dependencies.push(dep);
        self
    }

    pub fn new() -> Self {
        Self { attachments: vec![], subpasses: vec![], dependencies: vec![] }
    }

    pub fn build(&self, device: &Device) -> Result<RenderPass, ash::vk::Result> {
        let subpasses: Vec<vk::SubpassDescription> = self.subpasses.iter().map(|(bp, input, color, depth)|
            {
                let b = vk::SubpassDescription::builder()
                    .input_attachments(input.as_slice())
                    .color_attachments(color.as_slice())
                    .pipeline_bind_point(*bp);
                let b = if let Some(depth) = depth {
                    b.depth_stencil_attachment(depth)
                } else { b };
                b.build()
            }
        ).collect();
        let renderpass_create_info = vk::RenderPassCreateInfo::builder()
            .attachments(&self.attachments)
            .subpasses(&subpasses)
            .dependencies(&self.dependencies);

        unsafe {
            device.inner().create_render_pass(&renderpass_create_info, None)
        }.map(|raw| RenderPass { inner: Rc::new(RenderPassInner { raw, device: device.clone() }) })
    }
}
