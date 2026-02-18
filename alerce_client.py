"""
ALeRCE (Automatic Learning for the Rapid Classification of Events) API client.

Provides a clean interface for fetching ALeRCE classifier outputs and
cross-matching against TNS spectroscopic classifications.

ALeRCE broker: https://alerce.science/
ALeRCE API docs: https://alerce.readthedocs.io/

Notes
-----
ALeRCE uses a hierarchical classification scheme:
    - Stamp classifier: Real/Bogus → Stellar/Extragalactic (fast, alert-level)
    - Light curve classifier: Multi-class (Periodic / Stochastic / Transient)
    - Transient classifier: SN Ia, SN II, SN IIn, SN Ibc, TDE, AGN, CV/Nova, SLSN

Calibration audit targets the **transient classifier** outputs,
as these drive spectroscopic follow-up decisions.

References
----------
Sánchez-Sáez et al. (2021). Alert Classification for the ALeRCE Broker System:
    The Stamp Classifier. AJ 161, 141. https://arxiv.org/abs/2008.03312

Förster et al. (2021). The Automatic Learning for the Rapid Classification
    of Events (ALeRCE) Alert Broker. AJ 161, 242. https://arxiv.org/abs/2008.03303
"""

from __future__ import annotations

import time
import logging
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ALeRCE public API base URL
_ALERCE_API_BASE = "https://api.alerce.online/ztf/v1"

# Transient classifier class labels (ALeRCE v2 schema)
TRANSIENT_CLASSES = ["SNIa", "SNII", "SN IIn", "SNIbc", "TDE", "AGN", "CV/Nova", "SLSN"]

# Simplified 4-class mapping used in calibration audit
AUDIT_CLASSES = {"SNIa": 0, "SNII": 1, "TDE": 2, "AGN": 3}
AUDIT_CLASS_NAMES = ["SNIa", "SNII", "TDE", "AGN"]


class ALeRCEClient:
    """
    Client for the ALeRCE public API.

    Provides methods to:
        - Query classified objects with classifier probabilities
        - Cross-match against TNS spectroscopic classifications
        - Export clean DataFrames suitable for calibration analysis

    Parameters
    ----------
    base_url : str
        ALeRCE API base URL.
    rate_limit_delay : float
        Seconds to sleep between paginated API calls (be a good citizen).
    timeout : int
        HTTP request timeout in seconds.

    Examples
    --------
    >>> client = ALeRCEClient()
    >>> df = client.get_classified_objects(tns_crossmatch=True, n_objects=815)
    >>> df.head()
    """

    def __init__(
        self,
        base_url: str = _ALERCE_API_BASE,
        rate_limit_delay: float = 0.5,
        timeout: int = 30,
    ) -> None:
        self.base_url = base_url
        self.rate_limit_delay = rate_limit_delay
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "dac-calibration-audit/0.1"})

    def _get(self, endpoint: str, params: dict) -> dict:
        """Execute a GET request with error handling."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as e:
            logger.error(f"ALeRCE API HTTP error: {e} — URL: {url}")
            raise
        except requests.ConnectionError as e:
            logger.error(f"ALeRCE API connection error: {e}")
            raise

    def get_objects(
        self,
        classifier: str = "lc_classifier_transient",
        page_size: int = 100,
        n_objects: int = 1000,
        filters: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        Fetch classified objects from ALeRCE.

        Parameters
        ----------
        classifier : str
            Classifier type to query. Use "lc_classifier_transient" for
            the transient multi-class classifier.
        page_size : int
            Objects per API page (max 100).
        n_objects : int
            Maximum number of objects to retrieve.
        filters : dict, optional
            Additional query filters (e.g., {"classifierName": "SNIa"}).

        Returns
        -------
        pd.DataFrame
            Columns: oid, classifier probabilities, predicted class, ra, dec, etc.
        """
        records = []
        page = 1
        total_fetched = 0

        while total_fetched < n_objects:
            params = {
                "classifier": classifier,
                "page": page,
                "page_size": min(page_size, n_objects - total_fetched),
            }
            if filters:
                params.update(filters)

            try:
                data = self._get("/objects", params)
            except Exception:
                logger.warning(f"Failed on page {page}, stopping.")
                break

            items = data.get("items", [])
            if not items:
                break

            records.extend(items)
            total_fetched += len(items)
            page += 1
            time.sleep(self.rate_limit_delay)

        if not records:
            logger.warning("No objects returned from ALeRCE API.")
            return pd.DataFrame()

        df = pd.DataFrame(records)
        logger.info(f"Fetched {len(df)} objects from ALeRCE.")
        return df

    def get_probabilities(self, oid: str) -> dict:
        """
        Fetch detailed classifier probabilities for a single object.

        Parameters
        ----------
        oid : str
            ALeRCE object identifier (e.g., "ZTF21aaayiys").

        Returns
        -------
        dict
            Classifier probabilities for all classes.
        """
        data = self._get(f"/objects/{oid}/probabilities", {})
        return data

    def get_classified_objects(
        self,
        tns_crossmatch: bool = True,
        n_objects: int = 815,
        classes: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Fetch classifier outputs, optionally cross-matched with TNS.

        This is the primary method for the calibration audit. Returns a
        DataFrame with predicted probabilities and spectroscopic ground truth.

        Parameters
        ----------
        tns_crossmatch : bool
            If True, filter to objects with TNS spectroscopic classifications.
            This imposes a known selection bias (spectroscopically confirmed
            objects are brightness/priority biased), which should be acknowledged
            in any analysis.
        n_objects : int
            Target sample size. Default 815 matches the baseline audit sample.
        classes : list[str], optional
            Restrict to specific classes (e.g., ["SNIa", "SNII"]).

        Returns
        -------
        pd.DataFrame
            Columns include:
            - oid: ALeRCE object ID
            - prob_SNIa, prob_SNII, prob_TDE, prob_AGN: classifier probabilities
            - predicted_class: argmax class name
            - tns_class: spectroscopic classification (if tns_crossmatch=True)
            - tns_class_encoded: integer-encoded class label

        Notes
        -----
        The spectroscopic sample is **not** a random sample of ZTF transients.
        Objects with spectra tend to be brighter, more scientifically interesting,
        and more observationally accessible. This selection bias must be
        acknowledged when interpreting calibration metrics (see docs/methodology.md).
        """
        # NOTE: Full implementation requires ALeRCE + TNS cross-match pipeline.
        # The baseline audit was conducted using a pre-assembled dataset.
        # See notebooks/02_alerce_calibration_audit.ipynb for the full workflow.
        raise NotImplementedError(
            "Full API cross-match requires TNS credentials. "
            "See notebooks/02_alerce_calibration_audit.ipynb for the complete pipeline "
            "using the pre-assembled baseline audit dataset."
        )

    @staticmethod
    def encode_classes(labels: pd.Series, class_map: dict = AUDIT_CLASSES) -> np.ndarray:
        """
        Integer-encode string class labels.

        Parameters
        ----------
        labels : pd.Series of str class names
        class_map : dict mapping class name → integer

        Returns
        -------
        np.ndarray of int, with -1 for unknown classes.
        """
        encoded = labels.map(class_map).fillna(-1).astype(int)
        n_unknown = (encoded == -1).sum()
        if n_unknown > 0:
            logger.warning(f"{n_unknown} objects with unrecognised class labels → excluded.")
        return encoded.values


class TNSClient:
    """
    Client for the IAU Transient Name Server (TNS) API.

    Provides spectroscopic ground truth for calibration validation.
    TNS: https://www.wis-tns.org/

    Parameters
    ----------
    api_key : str
        TNS API key. Register at https://www.wis-tns.org/user/register.
    """

    _TNS_BASE = "https://www.wis-tns.org/api/get"

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "dac-calibration-audit/0.1"})

    def search(
        self,
        ra: float,
        dec: float,
        radius: float = 5.0,
        spectra_only: bool = True,
    ) -> list[dict]:
        """
        Cone search around (ra, dec) for TNS objects with spectroscopic classifications.

        Parameters
        ----------
        ra, dec : float
            Coordinates in degrees (J2000).
        radius : float
            Search radius in arcseconds.
        spectra_only : bool
            If True, return only objects with at least one classified spectrum.

        Returns
        -------
        list of dicts with TNS object info including classification type.
        """
        params = {
            "api_key": self.api_key,
            "ra": ra,
            "dec": dec,
            "radius": radius,
            "units": "arcsec",
            "objtype": "",
            "spectra": int(spectra_only),
        }
        response = self.session.get(
            f"{self._TNS_BASE}/search",
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("data", {}).get("reply", [])
