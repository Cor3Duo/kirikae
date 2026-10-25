use crate::cpu::Cpu;
use crate::services::stub::unimplemented;
use crate::services::{Reply, Request, Service};

// nn::prepo::detail::ipc::IPrepoService. Commands 10104/10105 are the [10.0.0-20.5.0]
// SaveReport/SaveReportWithUser variants (https://switchbrew.org/wiki/BCAT_services);
// both take an input buffer and have no output. Kirikae does not upload reports, so
// accepting them is enough.
pub struct Prepo;

impl Prepo {
    fn command(&self, req: &Request) -> Option<Reply> {
        match req.cmd {
            // SaveReport/SaveReportWithUser (all SDK revisions) and the system-report
            // variants: input buffers only, no output. Kirikae never uploads reports.
            10100..=10107 => Some(Reply::ok()),
            20100..=20103 => Some(Reply::ok()),
            // SetOperationMode and the storage/statistics clearing commands.
            20200 | 30100 | 30200 => Some(Reply::ok()),
            // RequestImmediateTransmission: no input, no output. Kirikae never uploads,
            // so the request completes and the status query stays idle.
            10200 => Some(Reply::ok()),
            // GetTransmissionStatus: u32 status, 0 (idle) when nothing was uploaded.
            10300 => Some(Reply::ok().u32(0)),
            _ => None,
        }
    }
}

impl Service for Prepo {
    fn name(&self) -> String {
        "prepo:u".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match self.command(req) {
            Some(r) => r,
            None => unimplemented("prepo:u", c, req),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_old_save_report_commands_without_output() {
        for cmd in [10104, 10105] {
            let mut req = Request {
                cmd,
                ..Default::default()
            };
            req.a.push(crate::services::Buf {
                addr: 0,
                size: 0x20,
            });
            req.x.push(crate::services::Buf {
                addr: 0,
                size: 0x40,
            });
            let r = Prepo.command(&req).expect("SaveReport command");
            assert!(r.data.is_empty());
            assert!(r.copy_handles.is_empty());
            assert!(r.move_handles.is_empty());
            assert!(r.objects.is_empty());
        }
    }

    #[test]
    fn immediate_transmission_has_no_output_and_status_is_idle() {
        let r = Prepo
            .command(&Request {
                cmd: 10200,
                ..Default::default()
            })
            .expect("RequestImmediateTransmission");
        assert_eq!(r.result, 0);
        assert!(r.data.is_empty());

        let r = Prepo
            .command(&Request {
                cmd: 10300,
                ..Default::default()
            })
            .expect("GetTransmissionStatus");
        assert_eq!(r.result, 0);
        assert_eq!(r.data, 0u32.to_le_bytes());
    }

    #[test]
    fn every_report_and_clear_command_is_accepted() {
        for cmd in [
            10100, 10101, 10102, 10103, 10106, 10107, 20100, 20101, 20102, 20103, 20200, 30100,
            30200,
        ] {
            let r = Prepo
                .command(&Request {
                    cmd,
                    ..Default::default()
                })
                .unwrap_or_else(|| panic!("command {cmd} is handled"));
            assert_eq!(r.result, 0, "command {cmd}");
            assert!(r.data.is_empty(), "command {cmd} has no output");
        }
    }

    #[test]
    fn unknown_commands_are_not_handled() {
        assert!(
            Prepo
                .command(&Request {
                    cmd: 19999,
                    ..Default::default()
                })
                .is_none()
        );
    }
}
