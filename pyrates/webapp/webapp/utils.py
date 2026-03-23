from __future__ import annotations

from typing import List, Optional, Any, Dict, Callable, cast
import io
import pandas as pd
import streamlit as st

# Centralized, one-time dependency initialization
def init_dependencies() -> Dict[str, Any]:
    """Initialize and cache optional dependencies once in Streamlit session state.

    Returns a dict with keys like:
      - has_rdkit: bool
      - Chem: rdkit.Chem module or None
      - has_ketcher: bool
      - st_ketcher: callable or None
    """
    if "_deps_init" in st.session_state and st.session_state.get("deps"):
        return st.session_state["deps"]

    # Detect RDKit
    try:
        from rdkit import Chem  # type: ignore
        has_rdkit = True
    except Exception:
        Chem = None  # type: ignore
        has_rdkit = False

    # Detect streamlit-ketcher
    try:
        from streamlit_ketcher import st_ketcher  # type: ignore
        has_ketcher = True
    except Exception:
        st_ketcher = None  # type: ignore
        has_ketcher = False

    deps = {
        "has_rdkit": has_rdkit,
        "Chem": Chem,
        "has_ketcher": has_ketcher,
        "st_ketcher": st_ketcher,
    }
    st.session_state["deps"] = deps
    st.session_state["_deps_init"] = True
    return deps


def render_molecule_drawer() -> Optional[Any]:
    """Render the molecule drawer if available; return raw representation or None.

    This function uses cached dependency info; if missing, it initializes once.
    """
    deps = st.session_state.get("deps") or init_dependencies()
    drawer = deps.get("st_ketcher")
    if callable(drawer):
        drawer_fn: Callable[[], Any] = cast(Callable[[], Any], drawer)
        return drawer_fn()  # may return SMILES str, MOL block str, or other structure
    else:
        st.info("Molecule drawer not available. Install 'streamlit-ketcher' to enable drawing.")
        return None


def _coerce_to_text(obj: Any) -> Optional[str]:
    """Best-effort conversion of various drawer outputs to text."""
    if obj is None:
        return None
    if isinstance(obj, bytes):
        for enc in ("utf-8", "latin-1"):
            try:
                return obj.decode(enc)
            except Exception:
                continue
        return None
    if isinstance(obj, (list, tuple)) and obj:
        # Many widgets return a tuple (smiles, something)
        return _coerce_to_text(obj[0])
    if isinstance(obj, dict):
        # Prefer explicit SMILES keys if present
        for k in ("smiles", "SMILES", "Smiles"):
            if k in obj and isinstance(obj[k], str) and obj[k].strip():
                return obj[k]
        # Otherwise, try MOL block-like keys
        for k in ("mol", "molblock", "molfile", "ctab"):
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                return v
        return None
    if isinstance(obj, str):
        return obj
    # Fallback: string representation if it's short
    try:
        s = str(obj)
        return s
    except Exception:
        return None


def convert_drawn_to_smiles(raw: Optional[Any]) -> Optional[str]:
    """Convert drawer output to SMILES using RDKit if available (cached deps).

    Accepts either SMILES directly or MOL blocks or other structures; returns canonical SMILES if possible.
    """
    text = _coerce_to_text(raw)
    if not text:
        return None
    s = text.strip()
    if not s:
        return None

    deps = st.session_state.get("deps") or init_dependencies()
    Chem = deps.get("Chem")
    if Chem is None:
        # Best effort: assume it's already SMILES
        return s

    # Heuristic: MOL blocks contain 'M  END' or multiple lines typical of V2000/3000
    if "M  END" in s or "\n" in s:
        try:
            mol = Chem.MolFromMolBlock(s, sanitize=True)
        except Exception:
            mol = None
        if mol is None:
            # Try as SMILES as a fallback
            try:
                mol = Chem.MolFromSmiles(s)
            except Exception:
                mol = None
        if mol is None:
            return None
        try:
            return Chem.MolToSmiles(mol)
        except Exception:
            return None

    # Try as SMILES
    try:
        mol = Chem.MolFromSmiles(s)
    except Exception:
        mol = None
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


# --------------- Existing helpers for parsing and gathering SMILES ---------------

def parse_smiles_multiline(text: str) -> List[str]:
    """Parse multiline text into a list of SMILES.

    - Splits on newlines
    - Strips whitespace
    - Skips empty lines and lines starting with '#'
    - If a line looks like CSV (contains comma), take the first token
    """
    if not text:
        return []
    smiles: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        token = line.split(",")[0].strip()
        if token:
            smiles.append(token)
    return smiles


def parse_uploaded_file(upload, smiles_column: str = "smiles") -> List[str]:
    """Parse uploaded .smi/.smiles/.txt or .csv file to extract SMILES."""
    if upload is None:
        return []

    name = getattr(upload, "name", "").lower()

    if name.endswith((".smi", ".smiles", ".txt")):
        try:
            upload.seek(0)
        except Exception:
            pass
        content = upload.read()
        try:
            text = content.decode("utf-8")
        except Exception:
            text = content.decode("latin-1", errors="ignore")
        return parse_smiles_multiline(text)

    if name.endswith(".csv"):
        try:
            upload.seek(0)
        except Exception:
            pass
        content = upload.read()
        df = pd.read_csv(io.BytesIO(content))
        # Pick the provided column name if present, otherwise common fallbacks, else first column
        candidates = [smiles_column, "smiles", "SMILES", "Smiles"]
        col = next((c for c in candidates if c in df.columns), df.columns[0])
        series = df[col].astype(str)
        return [s.strip() for s in series.tolist() if isinstance(s, str) and s.strip()]

    # Fallback: try plain text
    try:
        upload.seek(0)
    except Exception:
        pass
    try:
        content = upload.read()
        text = content.decode("utf-8")
    except Exception:
        return []
    return parse_smiles_multiline(text)


def gather_smiles(
    single_smiles: Optional[str] = None,
    drawn_smiles: Optional[str] = None,
    batch_text: Optional[str] = None,
    upload=None,
    smiles_column: str = "smiles",
) -> List[str]:
    """Gather and de-duplicate SMILES from the various inputs."""
    result: List[str] = []
    for s in [single_smiles or "", drawn_smiles or ""]:
        s = s.strip()
        if s:
            result.append(s)

    result.extend(parse_smiles_multiline(batch_text or ""))
    result.extend(parse_uploaded_file(upload, smiles_column=smiles_column))

    # Deduplicate while preserving order
    seen = set()
    uniq: List[str] = []
    for s in result:
        if s not in seen:
            uniq.append(s)
            seen.add(s)
    return uniq
