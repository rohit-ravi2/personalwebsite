import test from 'node:test';
import assert from 'node:assert/strict';
import { randomUUID } from 'node:crypto';
import handler from '../api/contact.mjs';
import { CONTACT_EMAIL, CONTACT_ORIGIN, validateSubmission, deliveryPayload, deliverMessage } from '../contact-lib/contact.mjs';
const valid = () => ({ name: 'A visitor', email: 'visitor@example.com', message: 'I would like to discuss a research collaboration.', topic: 'collaboration', website: '', submissionId: randomUUID() });
function request(body, options = {}) {
  return { method: options.method || 'POST', body, headers: { 'content-type': 'application/json', origin: CONTACT_ORIGIN, 'x-forwarded-for': options.ip || randomUUID(), ...options.headers }, socket: {} };
}
async function invoke(req) {
  const res = { code: 0, headers: {}, status(code) { this.code = code; return this; }, setHeader(k,v) { this.headers[k] = v; }, json(body) { this.body = body; return this; } };
  await handler(req,res); return res;
}
async function configured(fn) {
  const previous=process.env.RESEND_API_KEY;process.env.RESEND_API_KEY='test-only-not-a-real-key';
  try{return await fn();}finally{if(previous===undefined)delete process.env.RESEND_API_KEY;else process.env.RESEND_API_KEY=previous;}
}
test('contact validation rejects header injection, spam, invalid topics and oversized content', () => {
  assert.equal(validateSubmission(valid()).topic, 'collaboration');
  assert.equal(validateSubmission({ ...valid(), name: undefined }).name, '');
  for(const change of [{email:'visitor@example.com\nBcc: other@example.com'},{name:'Visitor\r\nBcc: other'},{message:'short'},{message:'x'.repeat(5001)},{website:'spam.example'},{topic:'__proto__'},{submissionId:'bad'}])assert.throws(()=>validateSubmission({...valid(),...change}));
  assert.throws(()=>validateSubmission(null));
});
test('delivery is addressed only to owner with a verified sender, reply address and provider idempotency', async () => configured(async()=>{
  const body=valid();const payload=deliveryPayload(body);
  assert.deepEqual(payload.to,[CONTACT_EMAIL]);assert.equal(payload.reply_to,body.email);
  assert.equal(payload.subject,'[rohitravi.com] Collaboration');assert.ok(payload.text.includes(body.message));
  let target;
  const result=await deliverMessage({...body,to:'other@example.com'},async(url,init)=>{target=url;assert.deepEqual(JSON.parse(init.body).to,[CONTACT_EMAIL]);assert.equal(init.headers['Idempotency-Key'],'contact/'+body.submissionId);return new Response(JSON.stringify({id:randomUUID()}));});
  assert.equal(target,'https://api.resend.com/emails');assert.equal(result.accepted,true);
}));
test('unconfigured or rejected delivery, malformed responses and timeouts never show success', async () => {
  const previous=process.env.RESEND_API_KEY;delete process.env.RESEND_API_KEY;
  try{await assert.rejects(deliverMessage(valid(),async()=>{throw Error('must not be called');}),error=>error.status===503);}finally{if(previous!==undefined)process.env.RESEND_API_KEY=previous;}
  await configured(async()=>{
    for(const fetcher of [async()=>new Response(JSON.stringify({name:'validation_error',message:'Domain not verified'}),{status:403}),async()=>new Response(JSON.stringify({error:'Rejected'})),async()=>new Response('not JSON',{status:500}),async()=>{throw new Error('timeout');}])await assert.rejects(deliverMessage(valid(),fetcher));
  });
});
test('endpoint enforces method, origin, content type and body limits before delivery', async () => {
  assert.equal((await invoke(request(valid(),{method:'GET'}))).code,405);
  assert.equal((await invoke(request(valid(),{headers:{origin:'https://other.example'}}))).code,403);
  assert.equal((await invoke(request(valid(),{headers:{'content-type':'text/plain'}}))).code,415);
  assert.equal((await invoke(request({...valid(),message:'x'.repeat(21000)}))).code,413);
  assert.equal((await invoke(request(null))).code,400);
});
test('successful retries and concurrent requests deliver once; changed payloads require a new reference', async () => configured(async()=>{
  const previous=globalThis.fetch;
  try{
    let calls=0;
    globalThis.fetch=async()=>{calls++;await new Promise(resolve=>setTimeout(resolve,10));return new Response(JSON.stringify({id:randomUUID()}));};
    const body=valid();const [first,second]=await Promise.all([invoke(request(body)),invoke(request(body))]);
    assert.equal(first.code,200);assert.equal(second.code,200);assert.equal(calls,1);assert.equal(first.body.reference,body.submissionId);
    assert.equal((await invoke(request({...body,message:'A different research question with the same reference.'}))).code,409);assert.equal(calls,1);
  }finally{globalThis.fetch=previous;}
}));
test('rate limiting and recoverable delivery errors preserve truthful responses', async () => configured(async()=>{
  const ip=randomUUID(), previous=globalThis.fetch;
  try{
    globalThis.fetch=async()=>new Response(JSON.stringify({error:'Temporary failure'}),{status:503});
    const failed=await invoke(request(valid()));assert.equal(failed.code,502);assert.equal(failed.body.accepted,undefined);
    for(let i=0;i<8;i++)await invoke(request({...valid(),message:'short'},{ip}));
    const limited=await invoke(request(valid(),{ip}));assert.equal(limited.code,429);assert.equal(limited.headers['Retry-After'],'900');
  }finally{globalThis.fetch=previous;}
}));

test('sender uses the domain provisioned by the mail integration', () => {
  const previousDomain=process.env.RESEND_EMAIL_DOMAIN, previousSender=process.env.CONTACT_FROM_EMAIL;
  try{
    delete process.env.CONTACT_FROM_EMAIL;process.env.RESEND_EMAIL_DOMAIN='rohitravi.com';
    assert.equal(deliveryPayload(valid()).from,'Rohit Ravi website <contact@rohitravi.com>');
    process.env.RESEND_EMAIL_DOMAIN='not/a/domain';
    assert.equal(deliveryPayload(valid()).from,'Rohit Ravi website <onboarding@resend.dev>');
  }finally{
    if(previousDomain===undefined)delete process.env.RESEND_EMAIL_DOMAIN;else process.env.RESEND_EMAIL_DOMAIN=previousDomain;
    if(previousSender===undefined)delete process.env.CONTACT_FROM_EMAIL;else process.env.CONTACT_FROM_EMAIL=previousSender;
  }
});
