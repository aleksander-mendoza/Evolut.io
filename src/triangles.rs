use crate::blocks::{World, Face};
use crate::blocks::block_properties::{BEDROCK, DIRT, GRASS, PLANK};
use ash::vk;
use crate::render::shader_module::ShaderModule;
use ash::vk::{ShaderStageFlags, PFN_vkCmdResetQueryPool};
use crate::render::pipeline::{PipelineBuilder, BufferBinding, PushConstant, Pipeline};
use crate::render::single_render_pass::SingleRenderPass;
use crate::render::submitter::Submitter;
use crate::render::command_pool::{CommandPool, CommandBuffer};
use failure::Error;
use crate::render::texture::{StageTexture, TextureView, Dim2D};
use crate::render::sampler::Sampler;
use crate::render::descriptor_layout::DescriptorLayout;
use crate::render::device::Device;
use crate::render::descriptors::{Descriptors, DescriptorsBuilder, DescriptorsBuilderLocked, UniformBufferBinding};
use crate::mvp_uniforms::MvpUniforms;
use crate::render::data::VertexClrTex;
use crate::render::stage_buffer::{StageBuffer, VertexBuffer, IndirectBuffer};
use crate::render::framebuffer::Framebuffer;
use crate::render::descriptor_pool::DescriptorSet;
use crate::render::swap_chain::SwapchainImageIdx;
use crate::display::Renderable;
use crate::render::imageview::Color;

pub struct TrianglesBuilder {
    pipeline: PipelineBuilder,
    texture: TextureView<Dim2D,Color>,
    sampler: Sampler,
    vertex_binding: BufferBinding<VertexClrTex>,
    descriptors:DescriptorsBuilderLocked,
    mvp_uniforms_binding:UniformBufferBinding<MvpUniforms,1>,
    vertex_buffer:VertexBuffer<VertexClrTex>,
    indirect:IndirectBuffer
}

impl TrianglesBuilder {
    pub fn new(cmd_pool:&CommandPool, mvp_uniforms:&MvpUniforms)->Result<Self,failure::Error>{
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
        let data = StageBuffer::new_vertex_buffer(cmd_pool, &data)?;
        let indirect = StageBuffer::new_indirect_buffer(cmd_pool, &[vk::DrawIndirectCommand{
            vertex_count: 3,
            instance_count: 4,
            first_vertex: 0,
            first_instance: 0
        }])?;
        let texture = StageTexture::new("assets/img/wall.jpg".as_ref(), cmd_pool, true)?;
        let indirect = indirect.take()?;
        let sampler = Sampler::new(cmd_pool.device(), vk::Filter::NEAREST, true)?;

        let frag = ShaderModule::new(include_glsl!("assets/shaders/blocks.frag", kind: frag) as &[u32], ShaderStageFlags::FRAGMENT, cmd_pool.device())?;
        let vert = ShaderModule::new(include_glsl!("assets/shaders/blocks.vert") as &[u32], ShaderStageFlags::VERTEX, cmd_pool.device())?;
        let texture = texture.take()?;
        let texture = texture.take();
        let data = data.take()?;
        let mut descriptors = DescriptorsBuilder::new();
        descriptors.sampler(&sampler,texture.imageview());
        let mvp_uniforms_binding = descriptors.singleton_uniform_buffer(mvp_uniforms);
        let descriptors = descriptors.make_layout(cmd_pool.device())?;

        let mut pipeline = PipelineBuilder::new();
        pipeline.descriptor_layout(descriptors.layout().clone())
            .shader("main", frag)
            .shader("main", vert)
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
            });
        let vertex_binding=  pipeline.vertex_input_from(0, data.gpu());

        Ok(Self{
            mvp_uniforms_binding,
            pipeline,
            vertex_binding,
            descriptors,
            sampler,
            vertex_buffer:data,
            indirect,
            texture
        })
    }

    pub fn create_pipeline(&mut self, render_pass:&SingleRenderPass) -> Result<TrianglesPipeline, Error> {
        let descriptors = self.descriptors.build(render_pass.swapchain())?;
        let pipeline = self.pipeline
            .reset_scissors()
            .scissors(render_pass.swapchain().render_area())
            .reset_viewports()
            .viewports(render_pass.swapchain().viewport())
            .build(render_pass)?;
        Ok(TrianglesPipeline {descriptors, pipeline})
    }

}

pub struct TrianglesPipeline {
    descriptors:Descriptors,
    pipeline:Pipeline
}

pub struct Triangles {
    compiled: TrianglesPipeline,
    builder: TrianglesBuilder
}

impl Triangles {
    const CLEAR_VALUES: [vk::ClearValue; 2] = [vk::ClearValue {
        color: vk::ClearColorValue {
            float32: [0.0, 0.0, 0.0, 1.0],
        },
    }, vk::ClearValue {
        depth_stencil: vk::ClearDepthStencilValue {
            depth: 1.,
            stencil: 0,
        },
    }];




    pub fn pipeline(&self) -> &Pipeline {
        &self.compiled.pipeline
    }
}

impl Renderable<MvpUniforms> for Triangles{
    fn new(cmd_pool: &CommandPool, render_pass: &SingleRenderPass, mvp_uniforms: &MvpUniforms) -> Result<Self, Error> {
        let mut builder = TrianglesBuilder::new(cmd_pool, mvp_uniforms)?;
        let compiled = builder.create_pipeline(render_pass)?;
        Ok(Self { builder, compiled })
    }
    fn record_cmd_buffer(&self, cmd: &mut CommandBuffer, image_idx:SwapchainImageIdx, render_pass:&SingleRenderPass)->Result<(),Error>{
        cmd
            .reset()?
            .begin(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE)?
            .render_pass(render_pass, render_pass.framebuffer(image_idx), render_pass.swapchain().render_area(), &Self::CLEAR_VALUES)
            .bind_pipeline(self.pipeline())
            .vertex_input(self.builder.vertex_binding,self.builder.vertex_buffer.gpu())
            .uniform(self.pipeline(), self.compiled.descriptors.descriptor_set(image_idx))
            .draw_indirect(self.builder.indirect.gpu())
            .end_render_pass()
            .end()?;
        Ok(())
    }
    fn update_uniforms(&mut self, image_idx:SwapchainImageIdx, mvp_uniform:&MvpUniforms){
        self.compiled.descriptors.uniform_as_slice_mut(image_idx,self.builder.mvp_uniforms_binding).copy_from_slice(std::slice::from_ref(mvp_uniform))
    }
    fn recreate(&mut self, render_pass: &SingleRenderPass) -> Result<(), Error> {
        self.compiled = self.builder.create_pipeline(render_pass)?;
        Ok(())
    }
}

