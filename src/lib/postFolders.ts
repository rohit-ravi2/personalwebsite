import type { CollectionEntry } from "astro:content";

type Post = CollectionEntry<"posts">;

// Display names for blog subfolders (src/content/posts/<folder>/...).
// A folder without an entry here gets a title derived from its name.
export const FOLDER_META: Record<
  string,
  { title: string; description: string }
> = {
  libribrain: {
    title: "LibriBrain",
    description:
      "Decoding heard words from MEG for the LibriBrain 2026 competition: the technical results, and how the ideas came together.",
  },
  "robotic-movement": {
    title: "Robotic Movement",
    description:
      "Notes on learning movement: running, jumping, carrying, and manipulation.",
  },
  "arc-prize-2026": {
    title: "ARC Prize 2026",
    description:
      "Notes on both ARC Prize 2026 tracks: what scored, what was only offline, and what is being submitted.",
  },
};

export const folderOf = (post: Post): string | null => {
  const i = post.slug.lastIndexOf("/");
  return i === -1 ? null : post.slug.slice(0, i);
};

export const folderTitle = (folder: string): string =>
  FOLDER_META[folder]?.title ??
  folder
    .split("/")
    .pop()!
    .split("-")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");

export const byDateDesc = (a: Post, b: Post) =>
  b.data.startDate.getTime() - a.data.startDate.getTime();

// Series order inside a folder: explicit `order` first, then oldest first.
export const bySeriesOrder = (a: Post, b: Post) =>
  (a.data.order ?? Infinity) - (b.data.order ?? Infinity) ||
  a.data.startDate.getTime() - b.data.startDate.getTime();

export const groupByFolder = (posts: Post[]) => {
  const root: Post[] = [];
  const folders = new Map<string, Post[]>();
  for (const p of posts) {
    const f = folderOf(p);
    if (f === null) root.push(p);
    else folders.set(f, [...(folders.get(f) ?? []), p]);
  }
  root.sort(byDateDesc);
  for (const list of folders.values()) list.sort(bySeriesOrder);
  return { root, folders };
};

export const cardHeading = (p: Post) =>
  p.data.draft ? `Draft: ${p.data.title}` : p.data.title;
