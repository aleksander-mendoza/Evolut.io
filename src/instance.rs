use std::ffi::CString;
use ash::version::{EntryV1_0, InstanceV1_0};
use crate::constants::APP_INFO;
use crate::platforms::{required_extension_names};
use ash::InstanceError;
use crate::validation_layer::{populate_debug_messenger_create_info, get_validation_layer_support};
use failure::err_msg;
use ash::vk::DebugUtilsMessengerCreateInfoEXT;

pub struct Instance {
    raw: ash::Instance,
    debug:Option<(ash::extensions::ext::DebugUtils,ash::vk::DebugUtilsMessengerEXT)>,
}

impl Instance {
    pub fn new(entry: &ash::Entry, debug:bool) -> Result<Self, failure::Error> {
        let mut extension_names = required_extension_names();
        let mut debug_builder = DebugUtilsMessengerCreateInfoEXT::builder();
        let mut instance_builder = ash::vk::InstanceCreateInfo::builder()
        .application_info(&APP_INFO)
        .enabled_extension_names(&extension_names);
        if debug{
            debug_builder = populate_debug_messenger_create_info(debug_builder);
            instance_builder = instance_builder.push_next(&mut debug_builder)
                .enabled_layer_names(get_validation_layer_support(entry)?);
        }
        let instance = unsafe { entry.create_instance(&instance_builder, None) }?;
        let debug_utils = if debug{
            let debug_utils_loader = ash::extensions::ext::DebugUtils::new(entry, &instance);
            let utils_messenger = unsafe { debug_utils_loader.create_debug_utils_messenger(&debug_builder, None)}?;
            Some((debug_utils_loader,utils_messenger))
        }else{
            None
        };
        Ok(Self{raw:instance,debug:debug_utils})
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        unsafe {
            if let Some((debug,messanger)) = self.debug.take(){
                debug.destroy_debug_utils_messenger(messanger,None);
            }
            self.raw.destroy_instance(None);
        }
    }
}