
#[repr(u8)]
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum FaceOrientation {
    XPlus = 0,
    XMinus = 1,
    YPlus = 2,
    YMinus = 3,
    ZPlus = 4,
    ZMinus = 5,
}

impl From<u8> for FaceOrientation{
    fn from(m: u8) -> Self {
       match m{
           0 => Self::XPlus,
           1 => Self::XMinus,
           2 => Self::YPlus,
           3 => Self::YMinus,
           4 => Self::ZPlus,
           5 => Self::ZMinus,
           t => panic!("Invalid enum {} for FaceOrientation",t)
       }
    }
}

impl FaceOrientation {
    pub fn from_dim(dim: usize, plus:bool) -> Self {
        Self::from((dim as u8)*2u8 + if plus{0u8}else{1u8})
    }
    pub fn is_side(&self) -> bool {
        (self.clone() as u8) > 1
    }
    pub fn is_plus(&self) -> bool {
        (self.clone() as u8) % 2 == 0
    }
    pub fn opposite(&self) -> FaceOrientation {
        assert_eq!(std::mem::size_of::<Self>(), std::mem::size_of::<u8>());
        let m = self.clone() as u8;
        unsafe {
            if m % 2 == 0 {
                std::mem::transmute(m + 1)
            } else {
                std::mem::transmute(m - 1)
            }
        }
    }
}
