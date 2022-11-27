use ash::vk;

use std::ffi::{CStr};
use std::os::raw::c_void;

use failure::err_msg;
use ash::vk::{LayerProperties};



pub fn layer_name(layer: &LayerProperties) -> &CStr {
    unsafe {
        CStr::from_ptr(layer.layer_name.as_ptr())
    }
}

pub unsafe extern "system" fn vulkan_debug_utils_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut c_void,
) -> vk::Bool32 {
    let severity = match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => "[Verbose]",
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => "[Warning]",
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => "[Error]",
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => "[Info]",
        _ => "[Unknown]",
    };
    let types = match message_type {
        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => "[General]",
        vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "[Performance]",
        vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "[Validation]",
        _ => "[Unknown]",
    };
    let message = CStr::from_ptr((*p_callback_data).p_message);
    println!("[Debug]{}{}{:?}", severity, types, message);

    vk::FALSE
}

const VK_LAYER_KHRONOS_VALIDATION: *const i8 = b"VK_LAYER_KHRONOS_validation\0".as_ptr() as *const i8;
const REQUIRED_VALIDATION_LAYERS: [*const i8; 1] = [VK_LAYER_KHRONOS_VALIDATION];

pub fn populate_debug_messenger_create_info(mut builder: vk::DebugUtilsMessengerCreateInfoEXTBuilder) -> vk::DebugUtilsMessengerCreateInfoEXTBuilder {
    builder.message_severity = vk::DebugUtilsMessageSeverityFlagsEXT::WARNING |
        // vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE |
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO |
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR;
    builder.message_type = vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION;
    builder.pfn_user_callback(Some(vulkan_debug_utils_callback))
}

pub fn get_validation_layer_support(entry: &ash::Entry) -> Result<&'static [*const i8], failure::Error> {
    // if support validation layer, then return true

    let layer_properties = entry.enumerate_instance_layer_properties()?;

    if layer_properties.len() <= 0 {
        return Err(err_msg("No available debug layers."));
    }

    let layer_properties: Vec<&CStr> = layer_properties.iter().map(layer_name).collect();


    for &required_layer_name in REQUIRED_VALIDATION_LAYERS.iter() {
        if !layer_properties.contains(&unsafe { CStr::from_ptr(required_layer_name) }) {
            return Err(err_msg("No available debug layers."));
        }
    }

    Ok(&REQUIRED_VALIDATION_LAYERS)
}
