#[macro_use]
extern crate vk_shader_macros;
extern crate nalgebra_glm as glm;
#[macro_use]
extern crate memoffset;

use crate::render::vulkan_context::VulkanContext;
use crate::display::{Display};
use std::ops::{Deref, DerefMut};
use crate::input::Input;
use crate::fps::FpsCounter;
use failure::err_msg;
use crate::blocks::{World, Block};
use crate::blocks::block_properties::{BEDROCK, DIRT, GRASS, PLANK};
use ash::vk;
use crate::render::command_pool::CommandBuffer;
use crate::render::framebuffer::Framebuffer;
use crate::render::descriptor_pool::DescriptorSet;
use crate::render::pipeline::Pipeline;
use crate::render::swap_chain::SwapChain;
use crate::render::render_pass::RenderPass;
use crate::render::submitter::Submitter;
use crate::mvp_uniforms::MvpUniforms;
// use crate::triangles::Triangles;
use crate::block_world::{BlockWorld, BlockWorldResources};
use crate::render::shader_module::ShaderModule;
use ash::vk::ShaderStageFlags;
use crate::particles::ParticleResources;
use crate::joint::{Joint, JointResources};
use crate::render::compute::ComputePipelineBuilder;
use crate::game::GameResources;

mod block_world;
mod display;
mod render;
mod input;
mod fps;
mod blocks;
mod mvp_uniforms;
mod triangles;
mod particle;
mod particles;
mod joint;
mod game;


fn main() -> Result<(), failure::Error> {
    run()
}

fn run() -> Result<(),failure::Error>{
    #[cfg(target_os = "macos")] {
        sdl2::hint::set("SDL_HINT_MAC_CTRL_CLICK_EMULATE_RIGHT_CLICK", "1");
    }
    let sdl = sdl2::init().map_err(err_msg)?;
    let video_subsystem = sdl.video().map_err(err_msg)?;
    let timer = sdl.timer().map_err(err_msg)?;
    let window = video_subsystem
        .window("Game", 900, 700)
        .vulkan()
        .resizable()
        .build()?;
    sdl.mouse().set_relative_mouse_mode(true);
    let mut  mvp_uniforms = MvpUniforms::new();
    let vulkan = VulkanContext::new(window)?;
    let mut display = Display::<MvpUniforms,GameResources>::new(vulkan,&mvp_uniforms)?;

    let event_pump = sdl.event_pump().map_err(err_msg)?;
    let mut input = Input::new(event_pump);
    let mut fps_counter = FpsCounter::new(timer, 60);
    let mut rotation = glm::quat_identity();
    let mut location = glm::vec3(2f32, 5f32, 2f32);
    let mut block_in_hand = Block::new(2u32);
    let model_matrix = glm::identity::<f32, 4>();
    let movement_speed = 0.005f32;
    let player_reach = 4f32;
    let rotation_speed = 1f32;
    let ash::vk::Extent2D { width, height } = display.extent();
    let mut projection_matrix = proj(width as f32, height as f32);

    display.rerecord_all_cmd_buffers()?;
    'main: loop {
        fps_counter.update();
        input.poll();
        let ash::vk::Extent2D { width, height } = display.extent();
        let (width, height) = (width as f32, height as f32);
        if input.quit() {
            break;
        }
        if input.escape() {
            input.reset_escape();
            sdl.mouse()
                .set_relative_mouse_mode(!sdl.mouse().relative_mouse_mode());
        }
        if input.has_mouse_move() {
            let normalized_x = (input.mouse_move_xrel() as f32) / width
                * fps_counter.delta_f32()
                * rotation_speed;
            let normalized_y = (input.mouse_move_yrel() as f32) / height
                * fps_counter.delta_f32()
                * rotation_speed;
            rotation = glm::quat_angle_axis(normalized_y, &glm::vec3(1f32, 0f32, 0f32))
                * rotation
                * glm::quat_angle_axis(normalized_x, &glm::vec3(0f32, 1f32, 0f32));
        }

        let movement_vector = input.get_direction_unit_vector() * movement_speed * fps_counter.delta_f32();
        let inverse_rotation = glm::quat_inverse(&rotation);
        let mut movement_vector = glm::quat_rotate_vec3(&inverse_rotation, &movement_vector);
        // display.pipeline().a().world().blocks().zero_out_velocity_vector_on_hitbox_collision(&mut movement_vector, &(location-glm::vec3(0.4f32,1.5,0.4)),&(location+glm::vec3(0.4f32,0.3,0.4)));
        location += movement_vector;
        if input.has_mouse_left_click() || input.has_mouse_right_click() {
            let ray_trace_vector = glm::vec4(0f32, 0., -player_reach, 0.);
            let ray_trace_vector = glm::quat_rotate_vec(&inverse_rotation, &ray_trace_vector);
            let world = display.pipeline_mut().block_world_mut().world_mut();
            if input.has_mouse_left_click() {
                world.ray_cast_remove_block(location.as_slice(), ray_trace_vector.as_slice());
            } else {
                world.ray_cast_place_block(location.as_slice(), ray_trace_vector.as_slice(), block_in_hand);
            }
            world.flush_all_chunks();
            world.reset();
            display.rerecord_all_cmd_buffers()?;
        }
        if input.number() > -1{
            block_in_hand = Block::new((input.number()+1) as u32)
        }
        let v = glm::quat_to_mat4(&rotation) * glm::translation(&-location);

        let m = model_matrix;

        mvp_uniforms.mv = &v * m;
        mvp_uniforms.mvp = projection_matrix * &mvp_uniforms.mv;
        if display.render(false,&mvp_uniforms)? {
            display.device().device_wait_idle()?;
            display.recreate()?;
            display.rerecord_all_cmd_buffers()?;
            projection_matrix = proj(width, height);
        }
    }
    display.device().device_wait_idle()?;
    Ok(())
}

fn proj(width: f32, height: f32) -> glm::Mat4 {
    let fov = 60f32 / 360f32 * std::f32::consts::PI * 2f32;
    glm::perspective(
        width / height,
        fov,
        0.1f32,
        200f32,
    )
}