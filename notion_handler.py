"""Notion integration — read RAWs URL, write XML URL, watch for status changes."""
from __future__ import annotations

import os
from notion_client import Client


def _get_client() -> Client:
    token = os.getenv("NOTION_TOKEN")
    if not token:
        raise RuntimeError("NOTION_TOKEN not set in environment")
    return Client(auth=token)


def get_page(page_id: str) -> dict:
    """Get a Notion page and extract the RAWs Drive URL."""
    client = _get_client()
    page = client.pages.retrieve(page_id=page_id)
    props = page.get("properties", {})

    # RAWs property — could be URL type or rich_text
    raws_url = None
    raws_prop = props.get("RAWs") or props.get("Raws") or props.get("raws")
    if raws_prop:
        prop_type = raws_prop.get("type")
        if prop_type == "url":
            raws_url = raws_prop.get("url")
        elif prop_type == "rich_text":
            texts = raws_prop.get("rich_text", [])
            if texts:
                raws_url = texts[0].get("plain_text") or texts[0].get("href")
        elif prop_type == "files":
            files = raws_prop.get("files", [])
            if files:
                raws_url = files[0].get("external", {}).get("url") or files[0].get("name")

    title_prop = props.get("Name") or props.get("Title") or props.get("title")
    title = ""
    if title_prop:
        title_items = title_prop.get("title", [])
        if title_items:
            title = title_items[0].get("plain_text", "")

    return {
        "page_id": page_id,
        "title": title,
        "raws_url": raws_url,
        "properties": props,
    }


def update_xml_property(page_id: str, drive_url: str):
    """Write the FCPXML Drive URL back to the Notion page's XML property."""
    client = _get_client()
    client.pages.update(
        page_id=page_id,
        properties={
            "XML": {"url": drive_url},
        },
    )
    print(f"  Notion updated: XML property set on {page_id}")


def get_ready_pages(db_id: str) -> list[dict]:
    """Query DB for pages with Status = 'Started Editing', RAWs filled, XML empty."""
    client = _get_client()

    results = client.databases.query(
        database_id=db_id,
        filter={
            "and": [
                {
                    "property": "Status",
                    "status": {"equals": "Started Editing"},
                },
                {
                    "property": "RAWs",
                    "url": {"is_not_empty": True},
                },
                {
                    "property": "XML",
                    "url": {"is_empty": True},
                },
            ]
        },
    )

    pages = []
    for page in results.get("results", []):
        pages.append(get_page(page["id"]))

    return pages
