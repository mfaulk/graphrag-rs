//! Terminal User Interface management
//!
//! Handles terminal initialization, cleanup, and event streaming.

use color_eyre::eyre::Result;
use crossterm::{
    event::{DisableMouseCapture, EnableMouseCapture, Event as CrosstermEvent, EventStream},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use futures::StreamExt;
use ratatui::{backend::CrosstermBackend, Terminal};
use std::io::{self, Stdout};
use tokio::{
    sync::mpsc::{self, UnboundedReceiver, UnboundedSender},
    task::JoinHandle,
    time::{self, Duration},
};
use tokio_util::sync::CancellationToken;

/// Event types from the terminal
#[derive(Debug, Clone)]
pub enum Event {
    /// Keyboard or mouse event from crossterm
    Crossterm(CrosstermEvent),
    /// Periodic tick for animations/updates
    Tick,
    /// Render frame
    Render,
    /// Terminal was resized
    Resize(u16, u16),
}

/// Terminal User Interface
pub struct Tui {
    /// The terminal instance
    pub terminal: Terminal<CrosstermBackend<Stdout>>,
    /// Background task handle
    task: JoinHandle<()>,
    /// Cancellation token for cleanup
    cancellation_token: CancellationToken,
    /// Event receiver
    event_rx: UnboundedReceiver<Event>,
    /// Event sender (for external use if needed)
    _event_tx: UnboundedSender<Event>,
    /// Frame rate (FPS)
    #[allow(dead_code)]
    frame_rate: f64,
    /// Tick rate (events per second)
    #[allow(dead_code)]
    tick_rate: f64,
}

impl Tui {
    /// Create a new TUI instance
    pub fn new() -> Result<Self> {
        let frame_rate = 60.0; // 60 FPS
        let tick_rate = 4.0; // 4 ticks per second

        let (event_tx, event_rx) = mpsc::unbounded_channel();
        let cancellation_token = CancellationToken::new();

        // Spawn event handler task
        let task = {
            let event_tx = event_tx.clone();
            let cancellation_token = cancellation_token.clone();
            let tick_duration = Duration::from_secs_f64(1.0 / tick_rate);
            let render_duration = Duration::from_secs_f64(1.0 / frame_rate);

            tokio::spawn(async move {
                let mut reader = EventStream::new();
                let mut tick_interval = time::interval(tick_duration);
                let mut render_interval = time::interval(render_duration);

                loop {
                    tokio::select! {
                        biased;

                        _ = cancellation_token.cancelled() => {
                            break;
                        }
                        maybe_event = reader.next() => {
                            match maybe_event {
                                Some(Ok(evt)) => {
                                    // Handle resize events specially
                                    if let CrosstermEvent::Resize(w, h) = evt {
                                        let _ = event_tx.send(Event::Resize(w, h));
                                    }
                                    let _ = event_tx.send(Event::Crossterm(evt));
                                }
                                Some(Err(_)) => {}
                                None => break,
                            }
                        }
                        _ = tick_interval.tick() => {
                            let _ = event_tx.send(Event::Tick);
                        }
                        _ = render_interval.tick() => {
                            let _ = event_tx.send(Event::Render);
                        }
                    }
                }
            })
        };

        // Initialize terminal
        let terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;

        Ok(Self {
            terminal,
            task,
            cancellation_token,
            event_rx,
            _event_tx: event_tx,
            frame_rate,
            tick_rate,
        })
    }

    /// Enter the alternate screen and enable raw mode
    pub fn enter(&mut self) -> Result<()> {
        enable_raw_mode()?;
        io::stdout().execute(EnterAlternateScreen)?;
        io::stdout().execute(EnableMouseCapture)?;
        self.terminal.hide_cursor()?;
        self.terminal.clear()?;
        Ok(())
    }

    /// Leave the alternate screen and disable raw mode.
    ///
    /// Shuts the event-reader task down *before* releasing the terminal so the
    /// background task cannot poll input against a terminal that is no longer
    /// in raw mode (which leaves the terminal in a corrupted state).
    pub async fn exit(&mut self) -> Result<()> {
        shutdown_event_task(&mut self.task, &self.cancellation_token).await;

        self.terminal.show_cursor()?;
        io::stdout().execute(DisableMouseCapture)?;
        io::stdout().execute(LeaveAlternateScreen)?;
        disable_raw_mode()?;
        Ok(())
    }

    /// Cancel the background task
    pub fn cancel(&self) {
        self.cancellation_token.cancel();
    }

    /// Get the next event
    pub async fn next(&mut self) -> Option<Event> {
        self.event_rx.recv().await
    }

    /// Get frame rate
    #[allow(dead_code)]
    pub fn frame_rate(&self) -> f64 {
        self.frame_rate
    }

    /// Get tick rate
    #[allow(dead_code)]
    pub fn tick_rate(&self) -> f64 {
        self.tick_rate
    }
}

impl Drop for Tui {
    fn drop(&mut self) {
        // Drop cannot await; do best-effort sync cleanup. Callers that care
        // about clean shutdown should call `exit().await` first.
        self.cancel();
        self.task.abort();
        let _ = self.terminal.show_cursor();
        let _ = io::stdout().execute(DisableMouseCapture);
        let _ = io::stdout().execute(LeaveAlternateScreen);
        let _ = disable_raw_mode();
    }
}

/// Signal cancellation and await the event task, falling back to abort if it
/// does not finish promptly. Returns once the task is no longer running.
async fn shutdown_event_task(task: &mut JoinHandle<()>, token: &CancellationToken) {
    if task.is_finished() {
        return;
    }
    token.cancel();
    let shutdown_grace = Duration::from_millis(200);
    if tokio::time::timeout(shutdown_grace, &mut *task)
        .await
        .is_err()
    {
        task.abort();
        // Best-effort: let the runtime observe the abort.
        let _ = (&mut *task).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    // shutdown_event_task awaits a cooperatively-cancellable task before
    // returning, so callers can rely on the task being done after the call.
    #[tokio::test]
    async fn shutdown_awaits_cooperative_task() {
        let token = CancellationToken::new();
        let observed_cancel = Arc::new(AtomicBool::new(false));

        let mut task = {
            let token = token.clone();
            let observed_cancel = Arc::clone(&observed_cancel);
            tokio::spawn(async move {
                token.cancelled().await;
                observed_cancel.store(true, Ordering::SeqCst);
            })
        };

        shutdown_event_task(&mut task, &token).await;

        assert!(
            task.is_finished(),
            "task should be joined before shutdown returns"
        );
        assert!(
            observed_cancel.load(Ordering::SeqCst),
            "task should observe cancellation before being joined",
        );
    }

    // shutdown_event_task aborts a task that ignores cancellation, so the
    // function never blocks the caller indefinitely.
    #[tokio::test]
    async fn shutdown_aborts_uncooperative_task() {
        let token = CancellationToken::new();

        let mut task = tokio::spawn(async {
            // Ignores cancellation; only an abort can stop this.
            std::future::pending::<()>().await;
        });

        shutdown_event_task(&mut task, &token).await;

        assert!(
            task.is_finished(),
            "task should be aborted after grace period"
        );
    }

    // shutdown_event_task is a no-op when the task has already terminated.
    #[tokio::test]
    async fn shutdown_is_idempotent_on_finished_task() {
        let token = CancellationToken::new();
        let mut task = tokio::spawn(async {});
        // Wait for the task to finish on its own.
        let _ = (&mut task).await;
        assert!(task.is_finished());

        shutdown_event_task(&mut task, &token).await;
        assert!(
            !token.is_cancelled(),
            "token should not be cancelled if task already finished"
        );
    }
}
