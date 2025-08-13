# View.py
from __future__ import annotations

import logging
from typing import List, Optional

try:
    import tkinter as tk
    from tkinter import StringVar
    from tkinter import ttk
except Exception:  # pragma: no cover
    # Fallback for very old environments; prefer Python 3 tkinter.
    from Tkinter import Tk as tk  # type: ignore
    from Tkinter import StringVar  # type: ignore
    import ttk  # type: ignore

from Utilities import CenterWindow

logger = logging.getLogger(__name__)


class View(ttk.Frame):
    """
    Tkinter UI for the Autocompleter.
    - Type to see suggestions (debounced).
    - Enter to force refresh & open dropdown.
    - Import button builds the trie from the configured file.
    - Switch button toggles Prefix-only vs Prefix+Infix.
    """

    def __init__(self, parent: "tk.Tk", controller) -> None:
        super().__init__(parent)
        self.parent = parent
        self.controller = controller

        self.parent.title("Intelligent Auto-Complete")
        self.pack(fill="both", expand=True, padx=8, pady=8)

        self._debounce_after_id: Optional[str] = None
        self._debounce_ms: int = 120  # small debounce for smoother typing

        self._init_components()
        CenterWindow(self.parent)

    # -----------------------
    # UI setup
    # -----------------------

    def _init_components(self) -> None:
        # Row container
        row = ttk.Frame(self)
        row.pack(fill="x", expand=True)

        # Input
        self.box_value = StringVar()
        # Debounced updates on text change (Python 3 uses trace_add)
        try:
            self.box_value.trace_add("write", self._on_text_changed)  # type: ignore[attr-defined]
        except Exception:
            # Back-compat for very old Tk
            self.box_value.trace("w", lambda *_: self._on_text_changed())

        self.box = ttk.Combobox(
            row,
            justify="left",
            width=50,
            textvariable=self.box_value,
            state="normal",
            values=[],
        )
        self.box.pack(side="left", fill="x", expand=True, padx=5, pady=5)
        self.box.bind("<<ComboboxSelected>>", self._on_combo_selected)
        self.box.bind("<Return>", self._on_return)

        # Import button
        self.import_button = ttk.Button(
            row,
            text="Import",
            command=self._on_import_clicked,
        )
        self.import_button.pack(side="left", padx=5)

        # Mode switch (prefix vs infix)
        self.cmd_str = StringVar(value="Prefix Only")
        self.switch_button = ttk.Button(
            row,
            textvariable=self.cmd_str,
            command=self._on_switch_clicked,
        )
        self.switch_button.pack(side="right", padx=5)

        # Status bar
        self.status = ttk.Label(self, text="Ready", anchor="w")
        self.status.pack(fill="x", padx=2, pady=(4, 0))

    # -----------------------
    # Event handlers
    # -----------------------

    def _on_text_changed(self, *_args) -> None:
        """Debounced updates as the user types."""
        if self._debounce_after_id is not None:
            try:
                self.after_cancel(self._debounce_after_id)
            except Exception:
                pass
        self._debounce_after_id = self.after(self._debounce_ms, self._refresh_suggestions)

    def _on_combo_selected(self, _event) -> None:
        """When a suggestion is picked from the dropdown."""
        # You can hook exact-match checks here if desired:
        # selected = self.box.get()
        # exists = self.controller.Contains(selected)
        # self._set_status(f"Selected: {selected}  (exists={exists})")
        pass

    def _on_return(self, _event) -> None:
        """Pressing Return refreshes suggestions and opens the dropdown."""
        self._refresh_suggestions(open_dropdown=True)

    def _on_import_clicked(self) -> None:
        """
        Import the word list and (re)build the trie.
        Shows a preview of the file contents in the dropdown.
        """
        try:
            preview: List[str] = self.controller.LoadFile()
            self.box["values"] = preview
            added = self.controller.Construct()
            self._set_status(f"Imported {added} entries.")
        except Exception as exc:
            logger.exception("Import failed: %s", exc)
            self._set_status(f"Import failed: {exc}")

    def _on_switch_clicked(self) -> None:
        """
        Toggle between 'Prefix Only' and 'Prefix and Infix'.
        """
        try:
            self.controller.SwitchCommand()
            curr = self.cmd_str.get()
            if curr == "Prefix Only":
                self.cmd_str.set("Prefix and Infix")
                self._set_status("Mode: Prefix + Infix")
            else:
                self.cmd_str.set("Prefix Only")
                self._set_status("Mode: Prefix Only")
            self._refresh_suggestions()
        except Exception as exc:
            logger.exception("Switch failed: %s", exc)
            self._set_status(f"Switch failed: {exc}")

    # -----------------------
    # Helpers
    # -----------------------

    def _refresh_suggestions(self, open_dropdown: bool = False) -> None:
        """Ask controller for suggestions and populate the combobox."""
        query = self.box_value.get()
        try:
            container = self.controller.List(query)
            self.box["values"] = container or []
            if open_dropdown and container:
                # Show the dropdown to the user
                self.box.event_generate("<Down>")
            self._set_status(f"{len(container)} suggestion(s)")
        except Exception as exc:
            logger.exception("Suggest failed: %s", exc)
            self._set_status(f"Suggest failed: {exc}")

    def _set_status(self, msg: str) -> None:
        self.status.configure(text=msg)