# pylint: disable=line-too-long
import os
from typing import Dict

from atlassian.confluence import Confluence
from dotenv import load_dotenv

load_dotenv(f"{os.path.dirname(__file__)}/../.env")  # config = {"USER": "foo", "EMAIL": "foo@example.org"}

if __name__ == "__main__":
    test_temp = """<ac:adf-extension><ac:adf-node type="panel"><ac:adf-attribute key="panel-type">note</ac:adf-attribute><ac:adf-content>
    <p>This template is brought to you by Muralaaaaaa, a visual collaboration app.</p></ac:adf-content></ac:adf-node><ac:adf-fallback>
    <div class="panel conf-macro output-block" style="background-color: rgb(234,230,255);border-color: rgb(153,141,217);border-width: 1.0px;">
    <div class="panelContent" style="background-color: rgb(234,230,255);">

    <p>This template is brought to you by Mural, a visual collaboration app.</p>
    </div></div></ac:adf-fallback></ac:adf-extension>
    <h2><ac:emoticon ac:name="blue-star" ac:emoji-shortname=":question:" ac:emoji-id="2753" ac:emoji-fallback="❓" /> <ac:inline-comment-marker ac:ref="8526d93f-957f-4825-a069-5c7014e1d96a">Unmet needs</ac:inline-comment-marker></h2>
    <p><ac:placeholder>Why are we doing this? What does the user expect to accomplish that they can't? Why is it important that we fix it for them?</ac:placeholder></p>
    <p />
    <h2><ac:emoticon ac:name="blue-star" ac:emoji-shortname=":bow_and_arrow:" ac:emoji-id="1f3f9" ac:emoji-fallback="🏹" /> Objectives</h2>
    <p><ac:placeholder>What do we want to accomplish with this project?</ac:placeholder></p>
    <p />
    <h2><ac:emoticon ac:name="blue-star" ac:emoji-shortname=":busts_in_silhouette:" ac:emoji-id="1f465" ac:emoji-fallback="👥" /> User personas</h2>
    <p><ac:placeholder>Which one of our personas is the main target for this change? You can use the Persona template to flesh these out.</ac:placeholder></p>
    <p />
    <h2><ac:emoticon ac:name="blue-star" ac:emoji-shortname=":muscle:" ac:emoji-id="1f4aa" ac:emoji-fallback="💪" /> Jobs we want to cover</h2>
    <ul>
    <li>
    <p><ac:inline-comment-marker ac:ref="f96966e8-32f7-4ce5-9dba-ada0e2d4a7e9">When I</ac:inline-comment-marker><ac:placeholder>&lt;describe the specific context&gt;</ac:placeholder></p></li>
    <li>
    <p>I want to<ac:placeholder>&lt;describe user need&gt;</ac:placeholder></p></li>
    <li>
    <p>So I can<ac:placeholder>&lt;describe the reason for the need&gt;</ac:placeholder></p></li></ul>
    <p />
    <h2><ac:emoticon ac:name="blue-star" ac:emoji-shortname=":scroll:" ac:emoji-id="1f4dc" ac:emoji-fallback="📜" /> Some history</h2>
    <p><ac:placeholder>If the issue has been previously explored, or there are other strategies we've tried here and learned from that have relevant learnings.</ac:placeholder></p>
    <p />
    <h2><ac:emoticon ac:name="blue-star" ac:emoji-shortname=":eyes:" ac:emoji-id="1f440" ac:emoji-fallback="👀" /> <ac:inline-comment-marker ac:ref="5a1f689c-959a-4dfb-a0e5-280994b3483f">Constraints</ac:inline-comment-marker></h2>
    <p><ac:placeholder>If this issue is affected by or affects another project.</ac:placeholder></p>
    <p />
    <h2><ac:emoticon ac:name="blue-star" ac:emoji-shortname=":map:" ac:emoji-id="1f5fa" ac:emoji-fallback="🗺" /> Explorations + Decisions</h2>
    <p><ac:placeholder>Outline what kinds of approaches were considered, and the benefits and drawbacks of each. What decisions did we make, and why?</ac:placeholder></p>
    <p />
    <h2><ac:emoticon ac:name="blue-star" ac:emoji-shortname=":calendar_spiral:" ac:emoji-id="1f5d3" ac:emoji-fallback="🗓" /> <ac:inline-comment-marker ac:ref="a732fa03-06f4-4df3-8700-e0a9a9e82660">Releases</ac:inline-comment-marker></h2>
    <p><ac:placeholder>If this project has releases, outline them here.</ac:placeholder></p>
    <table data-layout="default"><colgroup><col style="width: 136.0px;" /><col style="width: 136.0px;" /><col style="width: 136.0px;" /><col style="width: 136.0px;" /><col style="width: 136.0px;" /></colgroup>
    <tbody>
    <tr>
    <th>
    <p><strong>Release Name</strong></p></th>
    <th>
    <p><strong>Value it adds</strong></p></th>
    <th>
    <p><strong>Scope</strong></p></th>
    <th>
    <p><strong>Status</strong></p></th>
    <th>
    <p><strong>Completed date</strong></p></th></tr>
    <tr>
    <td>
    <p /></td>
    <td>
    <p /></td>
    <td>
    <p /></td>
    <td>
    <p><ac:structured-macro ac:name="status" ac:schema-version="1" ac:macro-id="18475d09-f549-402d-8aa5-0d1b31ccabf2"><ac:parameter ac:name="title">To do</ac:parameter></ac:structured-macro> / <ac:structured-macro ac:name="status" ac:schema-version="1" ac:macro-id="6ef6d88f-17d1-41e8-ba07-b52465c44c3a"><ac:parameter ac:name="title">In progress</ac:parameter><ac:parameter ac:name="colour">Blue</ac:parameter></ac:structured-macro> / <ac:structured-macro ac:name="status" ac:schema-version="1" ac:macro-id="ec3963e0-cb63-4cd1-a288-e35829fc33de"><ac:parameter ac:name="title">Blocked</ac:parameter><ac:parameter ac:name="colour">Red</ac:parameter></ac:structured-macro> / <ac:structured-macro ac:name="status" ac:schema-version="1" ac:macro-id="b3b32843-0c19-4b71-9fe1-08ebb7ee4991"><ac:parameter ac:name="title">Waiting for feedback</ac:parameter><ac:parameter ac:name="colour">Yellow</ac:parameter></ac:structured-macro> / <ac:structured-macro ac:name="status" ac:schema-version="1" ac:macro-id="9300ae4d-6e8f-464c-b43c-c3e87a575ae9"><ac:parameter ac:name="title">Done</ac:parameter><ac:parameter ac:name="colour">Green</ac:parameter></ac:structured-macro></p></td>
    <td>
    <p><time datetime="2021-09-12" /></p></td></tr></tbody></table>
    <h2><ac:emoticon ac:name="blue-star" ac:emoji-shortname=":footprints:" ac:emoji-id="1f463" ac:emoji-fallback="👣" /> Next steps</h2><ac:task-list>
    <ac:task>
    <ac:task-id>1</ac:task-id>
    <ac:task-status>incomplete</ac:task-status>
    <ac:task-body><span class="placeholder-inline-tasks"><ac:placeholder>What should we do next?</ac:placeholder><ac:structured-macro ac:name="status" ac:schema-version="1" ac:macro-id="cdddc2a2-9520-4062-a90f-cbca71f507be"><ac:parameter ac:name="title">Set a status</ac:parameter></ac:structured-macro></span></ac:task-body>
    </ac:task>
    </ac:task-list>
    <p />
    <h2><ac:emoticon ac:name="blue-star" ac:emoji-shortname=":chart_with_upwards_trend:" ac:emoji-id="1f4c8" ac:emoji-fallback="📈" /> Impact</h2>
    <p><ac:placeholder>Add in key metrics and other performance indicators that you are tracking.</ac:placeholder></p>
    <p />
    <h2><ac:emoticon ac:name="blue-star" ac:emoji-shortname=":link:" ac:emoji-id="1f517" ac:emoji-fallback="🔗" /> Other documents</h2>
    <ul>
    <li>
    <p><ac:placeholder>Example</ac:placeholder></p></li></ul>
    <p />
    <p />"""
    # Creating a new content template
    name = "test-name2"
    body = {"wiki": {"value": test_temp, "representation": "storage"}}
    template_type = "page"
    description = "test-desc"
    labels = None
    space = "JT"
    confluence = Confluence(
        url=os.environ["JIRA_URL"],
        username=os.environ["JIRA_USERNAME"],
        password=os.environ["JIRA_PASSWORD"],
        cloud=True,
    )
    confluence.url = os.environ["JIRA_URL"]  # bugfix
    # https://developer.atlassian.com/cloud/confluence/rest/api-group-template/#api-wiki-rest-api-template-put
    matching_pages = [t["templateId"] for t in confluence.get_content_templates(space) if t["name"] == name]
    template_id = None if len(matching_pages) == 0 else matching_pages[0]
    data: Dict[str, object] = {"name": name, "templateType": template_type, "body": body}

    if description:
        data["description"] = description

    if labels:
        data["labels"] = labels

    if space:
        data["space"] = {"key": space}

    if template_id:
        data["templateId"] = template_id
        print(confluence.put("wiki/rest/api/template", data=data))
    else:
        confluence.post("wiki/rest/api/template", json=data)
