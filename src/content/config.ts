// Import utilities from `astro:content`
import { z, defineCollection } from "astro:content";

// Define a `type` and `schema` for each collection
const projectCollection = defineCollection({
  type: "content",
  schema: z.object({
    title: z.string(),
    startDate: z.date(),
    description: z.string(),
    image: z
      .object({
        url: z.string(),
        alt: z.string(),
      })
      .optional(),
    tags: z.array(z.string()).optional(),
    subject: z.string(),
    overview: z
      .object({
        question: z.string().min(1).max(2000),
        contribution: z.string().min(1).max(2000),
        result: z.string().min(1).max(2000),
        links: z
          .array(
            z.object({
              label: z.string().min(1).max(100),
              href: z.string().regex(/^(https:\/\/[^\s]+|\/(?!\/)[^\s]*)$/),
            }),
          )
          .max(10),
      })
      .optional(),
    inProgress: z.boolean().optional(),
  }),
});

const experienceCollection = defineCollection({
  type: "content",
  schema: z.object({
    title: z.string(),
    startDate: z.date(),
    endDate: z.date().optional(),
    company: z.string(),
    tags: z.array(z.string()).optional(),
  }),
});

// ✅ Removed `slug` (Astro auto-generates it)

const postCollection = defineCollection({
  type: "content",
  schema: z.object({
    title: z.string(),
    startDate: z.date(),
    description: z.string(),
    image: z
      .object({
        url: z.string(),
        alt: z.string(),
      })
      .optional(),
    tags: z.array(z.string()).optional(),
    canonical: z.string().optional(),
    // Drafts are built and listed with a "Draft" marker, but kept off the home page.
    draft: z.boolean().optional(),
    // Optional position within a folder (series); lower comes first.
    order: z.number().optional(),
  }),
});

const certificationCollection = defineCollection({
  type: "content",
  schema: z.object({
    title: z.string(),
    issuer: z.string(),
    issueDate: z.date(),
    guidedProjects: z
      .array(
        z.object({
          group: z.string().min(1).max(150),
          title: z.string().min(1).max(300),
          description: z.string().min(1).max(1500),
          frameworks: z.array(z.string().min(1).max(100)).min(1).max(8),
          verifyUrl: z.string().url().startsWith("https://"),
          pdfPath: z
            .string()
            .regex(/^\/certifications\/[a-zA-Z0-9_-]+\.pdf$/)
            .optional(),
        }),
      )
      .min(1)
      .max(50)
      .optional(),
    courseCount: z.number().optional(),
    verifyUrl: z.string().optional(),
    pdfPath: z.string().optional(),
    tags: z.array(z.string()).optional(),
  }),
});

// Export a single `collections` object to register your collection(s)
export const collections = {
  projects: projectCollection,
  experiences: experienceCollection,
  posts: postCollection,
  certifications: certificationCollection,
};
