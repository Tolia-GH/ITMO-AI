(function () {
  "use strict";

  function readData(id) {
    var element = document.getElementById(id);
    if (!element) return [];

    try {
      return JSON.parse(element.textContent || "[]");
    } catch (error) {
      console.error("Unable to read page index data:", error);
      return [];
    }
  }

  function humanize(value) {
    return value
      .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
      .replace(/([A-Z])([A-Z][a-z])/g, "$1 $2")
      .replace(/([0-9])([A-Za-z])/g, "$1 $2")
      .replace(/[_-]+/g, " ")
      .trim();
  }

  function appendText(parent, tagName, className, text) {
    var element = document.createElement(tagName);
    if (className) element.className = className;
    element.textContent = text;
    parent.appendChild(element);
    return element;
  }

  function renderSubjects() {
    var container = document.querySelector("[data-subject-index]");
    if (!container) return;

    var subjects = readData("subject-index-data").map(function (subject) {
      var match = subject.slug.match(/^(\d+)(.+)$/);
      subject.termNumber = match ? Number(match[1]) : null;
      subject.displayName = humanize(match ? match[2] : subject.slug);
      subject.termLabel = match ? "第 " + match[1] + " 学期" : "未分类学科";
      return subject;
    });

    subjects.sort(function (left, right) {
      if (left.termNumber === null && right.termNumber !== null) return 1;
      if (right.termNumber === null && left.termNumber !== null) return -1;
      if (left.termNumber !== right.termNumber) {
        return (left.termNumber || 0) - (right.termNumber || 0);
      }
      return left.displayName.localeCompare(right.displayName);
    });

    var groups = new Map();
    subjects.forEach(function (subject) {
      if (!groups.has(subject.termLabel)) groups.set(subject.termLabel, []);
      groups.get(subject.termLabel).push(subject);
    });

    groups.forEach(function (items, termLabel) {
      var section = document.createElement("section");
      section.className = "term-section";
      appendText(section, "h3", "term-heading", termLabel);

      var grid = document.createElement("div");
      grid.className = "subject-grid";

      items.forEach(function (subject) {
        var link = document.createElement("a");
        link.className = "subject-card";
        link.href = subject.url;
        appendText(link, "span", "subject-card__term", termLabel);
        appendText(link, "strong", "", subject.displayName);
        appendText(link, "span", "subject-card__arrow", "浏览学科资料 →");
        grid.appendChild(link);
      });

      section.appendChild(grid);
      container.appendChild(section);
    });

    var count = document.querySelector("[data-index-count]");
    if (count) count.textContent = subjects.length + " 门学科";

    if (!subjects.length) {
      appendText(
        container,
        "p",
        "index-empty",
        "暂未发现包含 README.md 的学科目录。"
      );
    }
  }

  function createTreeNode(parent, name) {
    var node = parent.children.get(name);
    if (!node) {
      node = { name: name, children: new Map(), pages: [] };
      parent.children.set(name, node);
    }
    return node;
  }

  function compareNames(left, right) {
    return left.localeCompare(right, undefined, {
      numeric: true,
      sensitivity: "base"
    });
  }

  function buildTree(pages) {
    var root = { name: "", children: new Map(), pages: [] };

    pages.forEach(function (page) {
      var pathParts = (page.path || "")
        .replace(/\\/g, "/")
        .split("/")
        .filter(Boolean);
      if (!pathParts.length) return;

      var filename = pathParts.pop();
      var stem = filename.replace(/\.(?:md|html)$/i, "");
      var directories = pathParts;
      var logicalParts = directories.slice();
      var lastDirectory = directories[directories.length - 1];

      if (
        !lastDirectory ||
        lastDirectory.toLocaleLowerCase() !== stem.toLocaleLowerCase()
      ) {
        logicalParts.push(stem);
      }

      var node = root;
      logicalParts.forEach(function (part) {
        node = createTreeNode(node, part);
      });

      node.pages.push({
        title: page.title || humanize(stem),
        url: page.url,
        path: logicalParts.join("/")
      });
    });

    return root;
  }

  function renderTreeNodes(parent, nodes) {
    Array.from(nodes.values())
      .sort(function (left, right) {
        return compareNames(left.name, right.name);
      })
      .forEach(function (node) {
        var item = document.createElement("li");
        var children = Array.from(node.children.values());
        var hasGroup = children.length > 0 || node.pages.length > 1;
        var label =
          node.pages.length === 1
            ? node.pages[0].title
            : humanize(node.name);

        if (hasGroup) {
          item.className = "index-tree__group";
          appendText(item, "span", "index-tree__folder", label);

          if (node.pages.length) {
            var pagesList = document.createElement("ul");
            pagesList.className = "index-tree__pages";
            node.pages.forEach(function (page) {
              var pageItem = document.createElement("li");
              var pageLink = document.createElement("a");
              pageLink.href = page.url;
              pageLink.textContent =
                node.pages.length === 1 ? "打开此页面" : page.title;
              pageLink.className = "index-tree__page-link";
              pageItem.appendChild(pageLink);
              pagesList.appendChild(pageItem);
            });
            item.appendChild(pagesList);
          }

          if (children.length) {
            var nestedList = document.createElement("ul");
            nestedList.className = "index-tree__children";
            renderTreeNodes(nestedList, node.children);
            item.appendChild(nestedList);
          }
        } else if (node.pages.length === 1) {
          var link = document.createElement("a");
          link.href = node.pages[0].url;
          link.textContent = label;
          item.appendChild(link);
        }

        parent.appendChild(item);
      });
  }

  function renderContentIndex() {
    var container = document.querySelector("[data-content-index]");
    if (!container) return;

    var pages = readData("content-index-data");
    var treeList = container.querySelector("[data-index-tree]");
    var search = container.querySelector("[data-index-search]");
    var count = container.querySelector("[data-index-count]");
    var empty = container.querySelector("[data-index-empty]");

    function update(query) {
      var normalizedQuery = (query || "").trim().toLocaleLowerCase();
      var filtered = pages.filter(function (page) {
        return (
          !normalizedQuery ||
          [page.title, page.path]
            .join(" ")
            .toLocaleLowerCase()
            .indexOf(normalizedQuery) !== -1
        );
      });

      treeList.textContent = "";
      renderTreeNodes(treeList, buildTree(filtered).children);
      count.textContent = filtered.length + " 个页面";
      empty.hidden = filtered.length > 0;
    }

    if (search) {
      search.addEventListener("input", function () {
        update(search.value);
      });
    }

    update("");
  }

  renderSubjects();
  renderContentIndex();
})();
