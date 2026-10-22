const fs = require("node:fs/promises");
const path = require("node:path");
const { Notebook, matter, matterStringify } = require("crossnote");
const cheerio = require("cheerio");

const repositoryRoot = path.resolve(__dirname, "..");
const outputRoot = path.resolve(
  repositoryRoot,
  process.argv[2] || ".mpe-build/site-source"
);
const excludedEverywhere = new Set([".git", "node_modules"]);
const excludedRootDirectories = new Set([
  ".github",
  ".idea",
  ".mpe-build",
  ".vscode",
  "_site",
  "scripts",
  "venv"
]);
const excludedFiles = new Set(["package.json", "package-lock.json"]);
const siteTitle = "ITMO AI 学习资料";

function toPosixPath(value) {
  return value.split(path.sep).join("/");
}

function isExcluded(relativePath) {
  const parts = relativePath.split(path.sep);
  if (parts.some((part) => excludedEverywhere.has(part))) return true;
  return parts.length === 1 && excludedRootDirectories.has(parts[0]);
}

function isSubjectIndex(relativePath) {
  const parts = relativePath.split("/");
  return parts.length === 2 && parts[1].toLowerCase() === "readme.md";
}

function shouldRenderWithMpe(relativePath) {
  return (
    /\.md$/i.test(relativePath) &&
    relativePath.toLowerCase() !== "index.md" &&
    relativePath.toLowerCase() !== "404.md" &&
    !isSubjectIndex(relativePath)
  );
}

async function listMarkdownFiles(directory, result = []) {
  const entries = await fs.readdir(directory, { withFileTypes: true });

  for (const entry of entries) {
    const absolutePath = path.join(directory, entry.name);
    const relativePath = path.relative(repositoryRoot, absolutePath);
    if (
      entry.isDirectory() &&
      (isExcluded(relativePath) ||
        relativePath.split(path.sep)[0] === ".crossnote")
    ) {
      continue;
    }

    if (entry.isDirectory()) {
      await listMarkdownFiles(absolutePath, result);
    } else if (entry.isFile() && /\.md$/i.test(entry.name)) {
      result.push(absolutePath);
    }
  }

  return result;
}

async function copyProject(source, destination) {
  await fs.mkdir(destination, { recursive: true });
  const entries = await fs.readdir(source, { withFileTypes: true });

  for (const entry of entries) {
    const sourcePath = path.join(source, entry.name);
    const relativePath = path.relative(repositoryRoot, sourcePath);
    if (
      isExcluded(relativePath) ||
      (entry.isFile() && excludedFiles.has(entry.name))
    ) {
      continue;
    }

    await fs.cp(sourcePath, path.join(destination, entry.name), {
      recursive: true,
      filter(candidatePath) {
        const candidateRelativePath = path.relative(
          repositoryRoot,
          candidatePath
        );
        if (!candidateRelativePath) return true;
        if (isExcluded(candidateRelativePath)) return false;

        const parts = candidateRelativePath.split(path.sep);
        return !(
          parts.length === 1 &&
          excludedFiles.has(parts[0])
        );
      }
    });
  }
}

function encodeRelativeUrl(value) {
  return value
    .split("/")
    .map((part) => {
      if (part === "." || part === "..") return part;
      return encodeURIComponent(part);
    })
    .join("/");
}

function rewriteMarkdownHref(href, pagePath) {
  if (/^(?:[a-z][a-z\d+.-]*:|\/\/|#)/i.test(href)) return href;

  const suffixIndex = href.search(/[?#]/);
  const pathname = suffixIndex === -1 ? href : href.slice(0, suffixIndex);
  const suffix = suffixIndex === -1 ? "" : href.slice(suffixIndex);
  if (!/\.md$/i.test(pathname)) return href;

  const htmlPath = pathname.replace(/\.md$/i, ".html");
  if (pathname.startsWith("/")) {
    const repositoryPath = htmlPath.replace(/^\/+/, "");
    const relativePath = path.posix.relative(
      path.posix.dirname(pagePath),
      repositoryPath
    );
    return encodeRelativeUrl(relativePath) + suffix;
  }

  return htmlPath + suffix;
}

function addSiteNavigation(html, relativeHtmlPath, pageData) {
  const $ = cheerio.load(html);
  const parts = relativeHtmlPath.split("/");
  const subjectSlug = parts.length > 1 ? parts[0] : "";
  const pageDirectory = path.posix.dirname(relativeHtmlPath);
  const homeHref = encodeRelativeUrl(
    path.posix.relative(pageDirectory, "index.html") || "index.html"
  );
  const subjectReadmePath = subjectSlug
    ? path.posix.join(subjectSlug, "README.html")
    : "";
  const subjectHref = subjectReadmePath
    ? encodeRelativeUrl(
        path.posix.relative(pageDirectory, subjectReadmePath)
      )
    : "";

  const navigation = $("<nav>", {
    class: "mpe-site-navigation",
    "aria-label": "页面导航"
  });
  navigation.append(
    $("<a>", { href: homeHref }).text("← 返回主页")
  );

  if (
    subjectSlug &&
    fsSyncExists(path.join(repositoryRoot, subjectSlug, "README.md"))
  ) {
    navigation.append(
      $("<a>", { href: subjectHref }).text("返回学科目录")
    );
  }

  $("head").append(
    `<style>
      .mpe-site-navigation {
        box-sizing: border-box;
        position: sticky;
        top: 0;
        z-index: 1000;
        display: flex;
        gap: .75rem;
        align-items: center;
        padding: .7rem max(1rem, calc((100% - 960px) / 2));
        border-bottom: 1px solid #dce5e4;
        background: rgba(255, 255, 255, .96);
        font: 600 14px/1.4 Inter, "Segoe UI", sans-serif;
        backdrop-filter: blur(12px);
      }
      .mpe-site-navigation a {
        display: inline-block;
        padding: .45rem .7rem;
        border: 1px solid #dce5e4;
        border-radius: 8px;
        color: #075e5a;
        text-decoration: none;
      }
      .mpe-site-navigation a:hover {
        background: #e4f3ef;
      }
      @media (max-width: 640px) {
        .mpe-site-navigation { padding-inline: .75rem; }
      }
    </style>`
  );

  $("body").prepend(navigation);
  $("title").text(`${pageData.title} · ${siteTitle}`);

  $("a[href]").each((_, element) => {
    const href = $(element).attr("href");
    if (href) $(element).attr("href", rewriteMarkdownHref(href, relativeHtmlPath));
  });

  return $.html();
}

async function main() {
  await fs.rm(outputRoot, { recursive: true, force: true });
  await fs.mkdir(path.dirname(outputRoot), { recursive: true });
  await copyProject(repositoryRoot, outputRoot);

  const notebook = await Notebook.init({
    notebookPath: outputRoot,
    config: { enableScriptExecution: false }
  });
  const markdownFiles = await listMarkdownFiles(repositoryRoot);
  const renderFiles = markdownFiles.filter((file) =>
    shouldRenderWithMpe(toPosixPath(path.relative(repositoryRoot, file)))
  );

  for (const sourcePath of renderFiles) {
    const relativePath = toPosixPath(path.relative(repositoryRoot, sourcePath));
    const relativeHtmlPath = relativePath.replace(/\.md$/i, ".html");
    const stagedMarkdown = path.join(outputRoot, relativePath);
    const stagedHtml = path.join(outputRoot, relativeHtmlPath);
    const source = await fs.readFile(sourcePath, "utf8");
    const parsed = matter(source);
    const pageData = {
      ...parsed.data,
      title:
        parsed.data.title ||
        path.basename(relativePath, path.extname(relativePath)),
      layout: "mpe-content",
      mpe_content: true,
      render_with_liquid: false
    };

    // Render only in the staging copy; the repository's original HTML stays untouched.
    await fs.rm(stagedHtml, { force: true });
    const engine = notebook.getNoteMarkdownEngine(relativePath);
    const exportedPath = await engine.htmlExport({
      offline: false,
      runAllCodeChunks: false
    });
    const exportedHtml = await fs.readFile(exportedPath, "utf8");
    const finalHtml = addSiteNavigation(
      exportedHtml,
      relativeHtmlPath,
      pageData
    );

    await fs.mkdir(path.dirname(stagedHtml), { recursive: true });
    await fs.writeFile(
      stagedHtml,
      matterStringify(finalHtml, pageData),
      "utf8"
    );
    await fs.rm(stagedMarkdown, { force: true });
    console.log(`Rendered with MPE: ${relativePath}`);
  }

  console.log(
    `Prepared ${renderFiles.length} MPE-rendered pages in ${outputRoot}`
  );
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
