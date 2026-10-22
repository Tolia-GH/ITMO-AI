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

function prepareMpeDocument(html, relativeHtmlPath, pageData) {
  const $ = cheerio.load(html);
  // The Jekyll page owns navigation; Crossnote HTML is embedded in srcdoc.
  $("base").remove();
  $("head").prepend('<base target="_top">');
  $("title").text(`${pageData.title} · ITMO AI 学习资料`);

  $("a[href]").each((_, element) => {
    const href = $(element).attr("href");
    if (href) {
      $(element).attr("href", rewriteMarkdownHref(href, relativeHtmlPath));
      if (href.startsWith("#")) $(element).attr("target", "_self");
    }
  });

  return $.html();
}

async function main() {
  await fs.rm(outputRoot, { recursive: true, force: true });
  await fs.mkdir(path.dirname(outputRoot), { recursive: true });
  await copyProject(repositoryRoot, outputRoot);

  const notebook = await Notebook.init({ notebookPath: outputRoot });
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
    const finalHtml = prepareMpeDocument(
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
