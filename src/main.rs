#[macro_use]
extern crate vk_shader_macros;
extern crate nalgebra_glm as glm;
#[macro_use]
extern crate memoffset;

use crate::render::vulkan_context::VulkanContext;
use crate::display::Display;
use std::ops::{Deref, DerefMut};
use crate::input::Input;
use crate::fps::FpsCounter;
use failure::err_msg;

mod block_world;
mod display;
mod render;
mod input;
mod fps;
// mod blocks;

fn main() -> Result<(), failure::Error> {
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
    let vulkan = VulkanContext::new(window)?;
    let mut display = Display::new(vulkan)?;
    let event_pump = sdl.event_pump().map_err(err_msg)?;
    let mut input = Input::new(event_pump);
    let mut fps_counter = FpsCounter::new(timer,60);
    let mut rotation = glm::quat_identity();
    let mut location = glm::vec3(0f32, 0f32, 2f32);
    // let mut block_in_hand = Block::new(2u32);
    let model_matrix = glm::identity::<f32, 4>();
    let movement_speed = 0.005f32;
    let player_reach = 3f32;
    let rotation_speed = 1f32;
    let ash::vk::Extent2D { width, height } = display.extent();
    let mut projection_matrix = proj(width as f32, height as f32);
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
            let normalized_y = (-input.mouse_move_yrel() as f32) / height
                * fps_counter.delta_f32()
                * rotation_speed;
            rotation = glm::quat_angle_axis(normalized_y, &glm::vec3(1f32, 0f32, 0f32))
                * rotation
                * glm::quat_angle_axis(normalized_x, &glm::vec3(0f32, 1f32, 0f32));
        }

        let movement_vector = input.get_direction_unit_vector() * movement_speed * fps_counter.delta_f32();
        let inverse_rotation = glm::quat_inverse(&rotation);
        let mut movement_vector = glm::quat_rotate_vec3(&inverse_rotation, &movement_vector);
        // world.blocks().zero_out_velocity_vector_on_hitbox_collision(&mut movement_vector, &(location-glm::vec3(0.4f32,1.5,0.4)),&(location+glm::vec3(0.4f32,0.3,0.4)));
        location += movement_vector;

        let v = glm::quat_to_mat4(&rotation) * glm::translation(&-location);

        let m = model_matrix;
        let mv = &v * m;
        display.uniforms_mut().mvp = projection_matrix * &mv;
        if display.render()?{
            display.device().device_wait_idle()?;
            display.recreate()?;
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