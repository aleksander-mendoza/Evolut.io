use crate::render::instance::Instance;
use crate::render::device::{pick_physical_device, Device};
use crate::render::pipeline::{PipelineBuilder, Pipeline};
use crate::render::render_pass::{RenderPassBuilder, RenderPass};
use crate::render::shader_module::ShaderModule;
use ash::vk::{ShaderStageFlags, PhysicalDevice, ClearValue, VertexInputAttributeDescription};
use crate::render::command_pool::{CommandPool, CommandBuffer};
use crate::render::imageview::{ImageView, Color, Depth};
use crate::render::frames_in_flight::FramesInFlight;
use crate::render::surface::Surface;
use crate::render::swap_chain::SwapChain;
use crate::render::vulkan_context::VulkanContext;
use failure::Error;
use ash::vk;
use crate::render::data::{VertexClr, VertexClrTex, VertexSource};
use crate::render::buffer::{Buffer, StageBuffer, VertexBuffer, IndirectBuffer};
use crate::render::fence::Fence;
use crate::render::uniform_buffer::{ UniformBuffers};
use crate::render::descriptor_pool::{DescriptorPool, DescriptorSet};
use crate::render::descriptor_layout::DescriptorLayout;
use crate::render::texture::{StageTexture, Dim2D, TextureView};
use crate::render::sampler::Sampler;
use crate::render::framebuffer::Framebuffer;


struct DisplayInner<U:Copy> {
    // The order of all fields
    // is very important, because
    // they will be dropped
    // in the exact same order
    descriptor_sets: Vec<DescriptorSet>,
    descriptor_pool: DescriptorPool,
    framebuffers: Vec<Framebuffer>,
    pipeline: Pipeline,
    render_pass: RenderPass,
    depth_attachment: TextureView<Dim2D, Depth>,
    image_views: Vec<ImageView<Color>>,
    uniforms: UniformBuffers<U, 1>,
    swapchain: SwapChain,
}

pub struct DisplayData {
    pub data: VertexBuffer<VertexClrTex>,
    pub indirect: IndirectBuffer,
    texture: StageTexture<Dim2D>,
    sampler: Sampler
}

impl DisplayData{
    pub fn new(vulkan: &VulkanContext, cmd_pool: &CommandPool) -> Result<Self, failure::Error> {
        let data: [VertexClrTex; 3] = [
            VertexClrTex {
                pos: glm::vec2(0.0, -0.5),
                clr: glm::vec3(1.0, 0.0, 0.0),
                tex: glm::vec2(0.5, 1.0),
            },
            VertexClrTex {
                pos: glm::vec2(0.5, 0.5),
                clr: glm::vec3(0.0, 1.0, 0.0),
                tex: glm::vec2(1.0, 0.0),
            },
            VertexClrTex {
                pos: glm::vec2(-0.5, 0.5),
                clr: glm::vec3(0.0, 0.0, 1.0),
                tex: glm::vec2(0.0, 0.0),
            },
        ];
        let data = StageBuffer::new_vertex_buffer(vulkan.device(), cmd_pool, &data)?;
        let indirect = StageBuffer::new_indirect_buffer(vulkan.device(), cmd_pool, &[vk::DrawIndirectCommand{
            vertex_count: 3,
            instance_count: 4,
            first_vertex: 0,
            first_instance: 0
        }])?;
        let texture = StageTexture::new(vulkan.device(), "assets/img/wall.jpg".as_ref(), cmd_pool, true)?;
        let data = data.take()?;
        let texture = texture.take()?;
        let indirect = indirect.take()?;
        let sampler = Sampler::new(vulkan.device(), vk::Filter::NEAREST, true)?;
        Ok(Self { texture, data, sampler ,indirect})
    }
}



impl <U:Copy> DisplayInner<U> {
    pub fn new(vulkan: &VulkanContext, data: &DisplayData,uniform:U) -> Result<Self, failure::Error> {
        let swapchain = vulkan.instance().create_swapchain(vulkan.device(), vulkan.surface())?;
        let image_views = swapchain.create_image_views()?;
        let uniforms = UniformBuffers::new(vulkan.device(),&swapchain, uniform)?;
        let descriptor_layout = DescriptorLayout::new_sampler_uniform(&data.sampler, &uniforms)?;
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
            .build(&vulkan.device())?;
        let frag = ShaderModule::new(include_glsl!("assets/shaders/blocks.frag", kind: frag) as &[u32], ShaderStageFlags::FRAGMENT, vulkan.device())?;
        let vert = ShaderModule::new(include_glsl!("assets/shaders/blocks.vert") as &[u32], ShaderStageFlags::VERTEX, vulkan.device())?;
        let pipeline = PipelineBuilder::new()
            .descriptor_layout(descriptor_layout.clone())
            .shader("main", frag)
            .shader("main", vert)
            .scissors(swapchain.render_area())
            .viewports(swapchain.viewport())
            .vertex_input(0, data.data.gpu())
            .depth_test(true)
            .color_blend_attachment_states(vk::PipelineColorBlendAttachmentState {
                blend_enable: vk::FALSE,
                color_write_mask: vk::ColorComponentFlags::all(),
                src_color_blend_factor: vk::BlendFactor::ONE,
                dst_color_blend_factor: vk::BlendFactor::ZERO,
                color_blend_op: vk::BlendOp::ADD,
                src_alpha_blend_factor: vk::BlendFactor::ONE,
                dst_alpha_blend_factor: vk::BlendFactor::ZERO,
                alpha_blend_op: vk::BlendOp::ADD,
            })
            .build(&render_pass)?;
        let framebuffers: Result<Vec<Framebuffer>, vk::Result> = image_views.iter().map(|v| Framebuffer::new_color_and_depth(&render_pass, &swapchain, v, depth_attachment.imageview())).collect();
        let framebuffers = framebuffers?;

        let descriptor_pool = DescriptorPool::new(vulkan.device(), &descriptor_layout, &swapchain)?;
        let descriptor_sets = descriptor_pool.create_sets(&std::iter::repeat(descriptor_layout).take(swapchain.len()).collect::<Vec<DescriptorLayout>>())?;
        for (ds, u) in descriptor_sets.iter().zip(uniforms.buffers()) {
            ds.update_sampler(0, &data.sampler, data.texture.imageview());
            ds.update_buffer(1, u);
        }

        Ok(Self {
            depth_attachment,
            swapchain,
            image_views,
            framebuffers,
            render_pass,
            pipeline,
            uniforms,
            descriptor_pool,
            descriptor_sets,
        })
    }

}


pub struct Display<U:Copy> {
    command_buffers: Vec<CommandBuffer>,
    inner: DisplayInner<U>,
    data: DisplayData,
    cmd_pool: CommandPool,
    vulkan: VulkanContext,
}
impl <U:Copy> Display<U> {

    pub fn record_commands<F>(&mut self,f:F)->Result<(),vk::Result> where F:Fn(&mut CommandBuffer,&Framebuffer,&DescriptorSet,&Pipeline,&SwapChain,&RenderPass,&DisplayData)->Result<(), ash::vk::Result>{
        let Self{ command_buffers, inner, data, cmd_pool, vulkan } = self;
        let DisplayInner{ descriptor_sets, descriptor_pool, framebuffers,
            pipeline,
            render_pass, depth_attachment,
            image_views, uniforms, swapchain } = inner;
        for (cmd, (fb, ds)) in command_buffers.iter_mut().zip(framebuffers.iter().zip(descriptor_sets.iter())){
            f(cmd,fb,ds, pipeline,swapchain,render_pass,data)?
        }
        Ok(())
    }
    pub fn cmd_pool(&self) -> &CommandPool {
        &self.cmd_pool
    }
    pub fn device(&self) -> &Device {
        self.vulkan.device()
    }
    pub fn destroy(self) -> VulkanContext {
        let Self { vulkan, .. } = self;
        vulkan
    }
    pub fn swapchain(&self) -> &SwapChain {
        &self.inner.swapchain
    }
    pub fn extent(&self) -> vk::Extent2D {
        self.swapchain().extent()
    }
    pub fn uniforms_mut(&mut self)->&mut U{
        &mut self.inner.uniforms.data[0]
    }
    pub fn uniforms(&self)->&U{
        &self.inner.uniforms.data[0]
    }
    pub fn recreate(&mut self) -> Result<(), failure::Error> {
        self.inner = DisplayInner::new(&self.vulkan, &self.data, self.inner.uniforms.data[0])?;
        Ok(())
    }

    pub fn new(vulkan: VulkanContext, uniform:U) -> Result<Self, failure::Error> {
        let cmd_pool = CommandPool::new(vulkan.device(), true)?;
        let data = DisplayData::new(&vulkan, &cmd_pool)?;
        cmd_pool.reset()?;
        let inner = DisplayInner::new(&vulkan, &data, uniform)?;
        let command_buffers = cmd_pool.create_command_buffers(inner.framebuffers.len() as u32)?;
        Ok(Self {
            command_buffers,
            inner,
            vulkan,
            cmd_pool,
            data,
        })
    }
    pub fn render(&mut self) -> Result<bool, vk::Result> {
        let fence = self.vulkan.frames_in_flight().current_fence();
        fence.wait(None)?;
        let image_available = self.vulkan.frames_in_flight().current_image_semaphore();
        let (image_idx, is_suboptimal) = self.inner.swapchain.acquire_next_image(None, Some(image_available), None)?;
        let render_finished = self.vulkan.frames_in_flight().current_rendering();
        fence.reset()?;
        self.inner.uniforms.flush(image_idx as usize)?;
        let command_buffer = &self.command_buffers[image_idx as usize];
        command_buffer.submit(&[(image_available, vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)],
                              std::slice::from_ref(render_finished),
                              Some(fence))?;
        let result = self.inner.swapchain.present(std::slice::from_ref(render_finished), image_idx);
        let is_resized = match result {
            Ok(is_suboptimal) => is_suboptimal,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => true,
            Err(vk::Result::SUBOPTIMAL_KHR) => true,
            err => err?
        };
        self.vulkan.frames_in_flight_mut().rotate();
        Ok(is_suboptimal || is_resized)
    }
}