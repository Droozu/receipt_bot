from __future__ import annotations

import argparse
import json
import mimetypes
import os
from pathlib import Path
from typing import Any

import requests

# token = 'y0__xCu8vWnBhje1T4gkeuR1RYwsM_n6wdtLNXoy120DpcaRmPaFmUT1oInnw'

class YandexDiskAPIError(RuntimeError):
    pass


class YandexDiskClient:
    BASE_URL = "https://cloud-api.yandex.net/v1/disk"

    def __init__(self, oauth_token: str, timeout: int = 60) -> None:
        if not oauth_token.strip():
            raise ValueError("OAuth token is empty")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"OAuth {oauth_token}",
                "Accept": "application/json",
            }
        )

    def _check_response(self, response: requests.Response) -> dict[str, Any]:
        if response.ok:
            if response.content:
                try:
                    return response.json()
                except ValueError:
                    return {}
            return {}

        try:
            payload = response.json()
        except ValueError:
            payload = {"message": response.text}

        raise YandexDiskAPIError(
            f"Yandex Disk API error {response.status_code}: "
            f"{payload.get('message', 'unknown error')}"
        )

    def get_upload_link(self, remote_path: str, overwrite: bool = True) -> str:
        response = self.session.get(
            f"{self.BASE_URL}/resources/upload",
            params={
                "path": remote_path,
                "overwrite": str(overwrite).lower(),
            },
            timeout=self.timeout,
        )
        data = self._check_response(response)
        href = data.get("href")
        if not href:
            raise YandexDiskAPIError("Upload URL was not returned by API")
        return href

    def upload_file(self, local_file: str | Path, remote_path: str, overwrite: bool = True) -> None:
        local_path = Path(local_file)
        if not local_path.exists() or not local_path.is_file():
            raise FileNotFoundError(f"File not found: {local_path}")

        upload_url = self.get_upload_link(remote_path=remote_path, overwrite=overwrite)

        content_type, _ = mimetypes.guess_type(str(local_path))
        headers = {}
        if content_type:
            headers["Content-Type"] = content_type

        with local_path.open("rb") as f:
            response = requests.put(
                upload_url,
                data=f,
                headers=headers,
                timeout=self.timeout,
            )

        if not response.ok:
            raise YandexDiskAPIError(
                f"Upload failed with status {response.status_code}: {response.text}"
            )

    def publish(self, remote_path: str) -> None:
        response = self.session.put(
            f"{self.BASE_URL}/resources/publish",
            params={"path": remote_path},
            timeout=self.timeout,
        )
        self._check_response(response)

    def get_meta(self, remote_path: str, fields: str | None = None) -> dict[str, Any]:
        params = {"path": remote_path}
        if fields:
            params["fields"] = fields

        response = self.session.get(
            f"{self.BASE_URL}/resources",
            params=params,
            timeout=self.timeout,
        )
        return self._check_response(response)


    def get_preview(self, remote_path: str, size="L") -> str:
        """
        size: XS S M L XL XXL XXXL OR WxH
        """

        response = self.session.get(
            f"{self.BASE_URL}/resources",
            params={
                "path": remote_path,
                "fields": "preview",
                "preview_size": size,
                "preview_crop": "false",
            },
            timeout=self.timeout,
        )

        data = self._check_response(response)

        preview = data.get("preview")

        if not preview:
            raise YandexDiskAPIError("Preview not returned")

        return preview
    
    def upload_and_get_public_link(
        self,
        local_file: str | Path,
        remote_path: str,
        overwrite: bool = True,
        auto_publish: bool = True,
    ) -> dict[str, Any]:
        self.upload_file(local_file=local_file, remote_path=remote_path, overwrite=overwrite)

        if auto_publish:
            self.publish(remote_path)

        meta = self.get_meta(remote_path, fields="name,path,public_url,public_key,file, preview")
        public_url = meta.get("public_url")
        if not public_url:
            raise YandexDiskAPIError(
                "File uploaded, but public_url is empty. "
                "Check whether publishing succeeded."
            )

        return {
            "name": meta.get("name"),
            "path": meta.get("path"),
            "file_url": meta.get("file"),
            "public_url": public_url,
            "public_key": meta.get("public_key"),
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload file to Yandex Disk and get public link"
    )
    parser.add_argument("local_file", help="Path to local file")
    parser.add_argument(
        "--remote-path",
        required=True,
        help='Destination path on Yandex Disk, e.g. "disk:/uploads/receipt.jpg"',
    )
    parser.add_argument(
        "--token",
        default=os.getenv("YANDEX_DISK_TOKEN", ""),
        help="OAuth token. Can also be passed via YANDEX_DISK_TOKEN env var",
    )
    parser.add_argument(
        "--no-publish",
        action="store_true",
        help="Upload without publishing",
    )
    args = parser.parse_args()

    if not args.token:
        raise SystemExit(
            "OAuth token is required. Pass --token or set YANDEX_DISK_TOKEN"
        )

    client = YandexDiskClient(args.token)

    if args.no_publish:
        client.upload_file(args.local_file, args.remote_path)
    else:
        remote = args.remote_path

        client.upload_file(
            local_file=args.local_file,
            remote_path=remote,
        )

        client.publish(remote)

        preview = client.get_preview(remote)

        print("PREVIEW:", preview)


if __name__ == "__main__":
    main()
