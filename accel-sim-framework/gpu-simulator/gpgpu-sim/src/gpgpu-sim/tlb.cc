// Copyright (c) 2023-2024, Youngin Kim, William J. Song + Euijun Kim
// Yonsei University
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer;
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution;
// 3. Neither the names of Yonsei University nor the names of their
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "tlb.h"
#include "gpu-sim.h"
#include "shader.h"
#include "../../libcuda/gpgpu_context.h"

// Define TLB type
static const char *tlb_type_str[] = {
    "L1_i_TLB", "L1_d_TLB", "L2_u_TLB", "L3_u_TLB", "L4_u_TLB","NA"
};

tlb_hierarchy *m_tlb_hierarchy = nullptr;


bool tlb_entry_t::lookup(translation_request* tr, uint64_t time)
{
    if ((tr->addr) >> 19 == page_num)
    {
        accessed_time = time;
        return true;
    }
    else
        return false;
}

void tlb_entry_t::entry_flush()
{
    page_num = 1;
    accessed_time = 1;
}


unsigned set_t::m_next_set_id = 0;
set_t::set_t(unsigned num_entries, unsigned num_sub_entries, tlb_t* tlb,
             tlb_hierarchy* tlb_hierarchy)
  : m_set_id(m_next_set_id++),
    m_num_entries(num_entries),
    m_entries(num_entries, tlb_entry_t(num_sub_entries, tlb_hierarchy)),
    m_tlb(tlb),
    m_tlb_hierarchy(tlb_hierarchy)
{
    //std::flush();
}

void set_t::insert_tlb_entry(translation_request* tr, uint64_t time)
{
    unsigned last_accessed_idx = 0;
    uint64_t last_accessed_time = m_entries[0].accessed_time;

    // Find if empty entry exists, update it
    for (unsigned i = 0; i < m_num_entries; i++)
    {
        if (m_entries[i].accessed_time == 1)
        {
            m_entries[i].page_num = (tr->addr) >> 19;
            m_entries[i].accessed_time = time;
            return;
        }

        if(last_accessed_time > m_entries[i].accessed_time)
        {
            last_accessed_idx = i;
            last_accessed_time = m_entries[i].accessed_time;
        }
    }

    // Remove oldest entry and update with new access data
    m_entries[last_accessed_idx].page_num = (tr->addr) >> 19;
    m_entries[last_accessed_idx].accessed_time = time;
}


bool set_t::lookup(translation_request* tr, uint64_t time)
{
    // If lookup is done on L4 TLB (PTW it is always hit)
    if (m_tlb->get_tlb_type() == L4_u_TLB)
        return true;

    // Other TLB layers
    uint64_t page_number_for_print = (tr->addr)>>19;

    // Loop through ways in a set
    for (unsigned i = 0; i < m_num_entries; i++)
    {
        if (m_entries[i].lookup(tr, time))  // If hit, time update is done on "lookup" function
        {
            // printf("TLB[%d] hit! %llu\n", m_tlb->get_tlb_id(), (unsigned long long)time);
            return true;
        }
    }

    // If miss
    // printf("TLB[%d] miss! %llu\n", m_tlb->get_tlb_id(), (unsigned long long)time);

    return false;
}

void set_t::print_tlb_set()
{
    for (unsigned i=0; i<m_entries.size(); i++)
    {
        tlb_entry_t* current_entry = &m_entries[i];
        printf("%llu ", (unsigned long long)current_entry->page_num);
    }
}

void set_t::print_tlb_entry_time()
{
    for (unsigned i=0; i<m_entries.size(); i++)
    {
        if (i == m_num_entries/2)
            printf("\n");

        tlb_entry_t* current_entry = &m_entries[i];
        printf("%llu(%llu) ", (unsigned long long)current_entry->page_num, (unsigned long long)current_entry->accessed_time);
    }
    printf("\n");
}

bool set_t::check_page_exist(uint64_t page_num)
{
    for (unsigned i=0; i<m_entries.size(); i++)
    {
        tlb_entry_t* current_entry = &m_entries[i];
        if(current_entry->page_num == page_num)
            return true;
    }

    return false;
}

// Type 1: tlb_t
tlb_t::tlb_t(tlb_type type, unsigned tlb_id, tlb_hierarchy* tlb_hierarchy)
  : m_type(type), m_tlb_id(tlb_id), m_tlb_hierarchy(tlb_hierarchy)
    // m_gpu(m_tlb_hierarchy->get_gpu())
{
    switch (m_type) {
        case L1_i_TLB:
            m_access_latency = 1;
            break;
        
        case L1_d_TLB:
            m_access_latency = 10;
            break;

        case L2_u_TLB:
            m_access_latency = 40;
            break;

        case L3_u_TLB:
            m_access_latency = 120;
            break;

        case L4_u_TLB:
            m_access_latency = 700;
            break;

        default:
            assert(0);
            break;
    }
}

unsigned tlb_t::get_tlb_id()
{
    return m_tlb_id;
}

tlb_type tlb_t::get_tlb_type()
{
    return m_type;
}

unsigned tlb_t::get_translation_queue_size()
{
    return m_translation_queue.size();
}

// Type 2: u_tlb_t
u_tlb_t::u_tlb_t(tlb_type type, unsigned tlb_id, tlb_hierarchy* tlb_hierarchy)
  : tlb_t(type, tlb_id, tlb_hierarchy)
{
    switch(type) {

        case L1_i_TLB:
            assert(0);
            break;

        case L1_d_TLB:
            m_num_sets = 1;
            m_num_ways = 16;
            m_num_sub_entries = 1;
            break;
        
        case L2_u_TLB:
            m_num_sets = 256;
            m_num_ways = 8;
            m_num_sub_entries = 1;
            break;

        case L3_u_TLB:
            m_num_sets = 512;
            m_num_ways = 8;
            m_num_sub_entries = 1;
            break;
        
        case L4_u_TLB:
            m_num_sets = 1;
            m_num_ways = 1;
            m_num_sub_entries = 1;
            break;

        default:
            assert(0);
            break;
    }
    
    for (unsigned i = 0; i < m_num_sets; i++)
        m_sets.push_back(new set_t(m_num_ways, m_num_sub_entries, this, tlb_hierarchy));
}

void u_tlb_t::cycle(uint64_t time)
{
    access(time);
    commit(time);
    issue_missed_translations(time);
}

unsigned u_tlb_t::get_set(uint64_t addr)
{
    switch(m_type)
    {
        case L1_d_TLB:
            return 0;

        case L2_u_TLB:
        case L3_u_TLB: {
            uint64_t input_page_num = addr >> 19;
            uint64_t set_index = input_page_num & (m_num_sets - 1);
            return set_index;
        }

        case L4_u_TLB:
            return 0;

        default:
            assert(0);
            break;
    }
}

void u_tlb_t::fill(translation_request* tr, uint64_t time)
{
    unsigned set_index = get_set(tr->addr);
    m_sets[set_index]->insert_tlb_entry(tr, time);

    if (check_merged_page_exist((tr->addr)>>19, time))
        m_merged_pages.remove((tr->addr)>>19);
    m_completed_translations.push_back(tr);
}

bool u_tlb_t::check_merged_page_exist(uint64_t page_num, uint64_t time)
{
    std::list<uint64_t>::iterator it;
    for (it = m_merged_pages.begin(); it != m_merged_pages.end(); ++it)
    {
        if (page_num == *it)
            return true;
    }

    /*
    std::list<uint64_t> tmp = m_merged_pages;
    while (!tmp.empty())
    {
        uint64_t page = tmp.front();
        if (page_num == page)
            return true;
        tmp.pop_front();               // remove the element we just processed
    }
    */

    return false;
}

void u_tlb_t::access(uint64_t time)
{
    if(m_translation_queue.empty() && m_merged_translations.empty())
        return;

    // Print TLB stat
    /*
    if (m_type == L1_d_TLB && m_tlb_id == 51)
    {
        if (time > 298920 && time < 299020)
        {
            printf("Calls print_all_for_debugging!\n");
            print_all_for_debugging(time);

            if (time > 298920 && time < 299020)
                print_merged_pages_list();

            fflush(stdout);
        }
    }
    */

    // Get frontmost tr from merged_translation queue
    if (m_type == L1_d_TLB || m_type == L2_u_TLB || m_type == L3_u_TLB)
    {
        if (!m_merged_translations.empty())
        {
            translation_request* tr_merged = m_merged_translations.front();
            uint64_t access_addr_merged = tr_merged->addr;
            uint64_t addr_page_merged = access_addr_merged >> 19;
            unsigned set_index_merged = get_set(access_addr_merged);

            if (m_sets[set_index_merged]->lookup(tr_merged, time))  // Outstanding hit
            {
                m_merged_translations.pop();
                if (check_merged_page_exist(addr_page_merged, time))
                {
                    m_merged_pages.remove(addr_page_merged);
                    // if (time > 298920 && time < 299020 && m_tlb_id == 51)
                    //    printf("m_merged_pages.remove(%llu)\n", (unsigned long long)addr_page_merged);
                }
                // else if (time > 298920 && time < 299020 && m_tlb_id == 51)
                //    printf("m_merged_pages.remove(%llu)\n", (unsigned long long)addr_page_merged);

                m_completed_translations.push_back(tr_merged);
                return;
            }
            else if (!check_merged_page_exist(addr_page_merged, time))
            {
                // Miss data (merged) not on TLB
                // Miss data (merged) not on m_merged_pages
                // It means that this m_merged_translations will never be popped.
                // It should have been evicted on TLB for some reason
                printf("[Warning] merged_tr not on TLB, not on m_merged_pages\n");
                fflush(stdout);
                assert(0);
            }
        }

        if (!m_translation_queue.empty())
        {
            translation_request* tr = m_translation_queue.front();
            if (time >= tr->m_mf->get_status_change() + m_access_latency)
            {
                m_translation_queue.pop();

                uint64_t access_addr = tr->addr;
                uint64_t addr_page = access_addr >> 19;
                unsigned set_index = get_set(access_addr);

                // If TLB hit
                if (m_sets[set_index]->lookup(tr, time))
                {
                    m_completed_translations.push_back(tr);
                }
                else // If TLB miss
                {
                    // But exists on m_merged_pages
                    if(check_merged_page_exist(addr_page, time))
                        m_merged_translations.push(tr);
                    else  // Not exist at all
                    {
                        m_merged_pages.push_back(addr_page);  // tr accessing same pages will enter different queue

                        // if (m_tlb_id == 51 && addr_page == 66775061)
                        //    printf("Time:%llu - crmininal page enters m_merged_pages\n", (unsigned long long)time);

                        m_pending_missed_translations.push(tr);
                    }
                }
            }
        }
    }
    else if (!m_translation_queue.empty())  // Just for page table walk
    {
        translation_request* tr = m_translation_queue.front();
        if (time >= tr->m_mf->get_status_change() + m_access_latency)
        {
            m_translation_queue.pop();

            uint64_t access_addr = tr->addr;
            uint64_t addr_page = access_addr >> 19;
            unsigned set_index = get_set(access_addr);

            // If TLB hit
            if (m_sets[set_index]->lookup(tr, time))
            {
                m_completed_translations.push_back(tr);
            }
            else // If TLB miss
            {
                // But exists on m_merged_pages
                if(check_merged_page_exist(addr_page, time))
                    m_merged_translations.push(tr);
                else  // Not exist at all
                    m_pending_missed_translations.push(tr);
            }
        }
    }
}

void u_tlb_t::commit(uint64_t time)
{
    if (m_completed_translations.empty()) return;
    translation_request* tr = m_completed_translations.front();

    // Commit TLB hit mfs
    if (m_tlb_hierarchy->can_commit_translation(m_type, tr))
    {
        m_completed_translations.pop_front();
        m_tlb_hierarchy->commit_translation(time, m_type, tr);
    }
}


void u_tlb_t::issue_missed_translations(uint64_t time)
{
    if (m_pending_missed_translations.empty()) return;
    translation_request* tr = m_pending_missed_translations.front();

    if (m_tlb_hierarchy->tlb_miss(time, m_type, m_tlb_id, tr))
    {
        m_pending_missed_translations.pop();
    }
}

void u_tlb_t::push_to_complete_queue(translation_request* tr)
{
    m_completed_translations.push_back(tr);
}


// Type 3: d_tlb_t
d_tlb_t::d_tlb_t(tlb_type type, unsigned tlb_id, tlb_hierarchy* tlb_hierarchy)
  : u_tlb_t(type, tlb_id, tlb_hierarchy),
    m_stalled(false)
{
}

void d_tlb_t::cycle(uint64_t time)
{
    access(time);
    commit(time);
    issue_missed_translations(time);
}


tlb_hierarchy::tlb_hierarchy(const shader_core_config *shader_config, class gpgpu_sim *gpu)
: m_shader_config(shader_config), m_gpu(gpu)
{ 
    // L1 TLB
    m_n_l1_tlb = 54;
    for (unsigned i = 0; i < m_n_l1_tlb; i++)
        m_l1_d_tlb.push_back(new d_tlb_t(L1_d_TLB, i, this));

    // L2 uTLB
    m_n_l2_tlb = 7;
    for (unsigned i = 0; i < m_n_l2_tlb; i++)
        m_l2_u_tlb.push_back(new u_tlb_t(L2_u_TLB, i, this));

    // L3 TLB
    m_l3_u_tlb = new u_tlb_t(L3_u_TLB, 0, this);

    // L4 TLB
    m_l4_u_tlb = new u_tlb_t(L4_u_TLB, 0, this);

}

void tlb_hierarchy::push_translation_request(uint64_t time, mem_fetch* mf)
{
    translation_request* tr = new translation_request(time, mf);

    mf->set_status(IN_L1TLB_ACCESS_QUEUE, time);
    
    // Instruction related stuff (), push it directly to icnt
    if (mf->get_access_type() == INST_ACC_R || mf->get_access_type() == CONST_ACC_R)
    {
        // just apply 1 cycle for CONST_ACC_R too
        m_gpu->icnt_inject_request_packet(tr->m_mf);
    } 
    else if (mf->get_access_type() == GLOBAL_ACC_R || mf->get_access_type() == GLOBAL_ACC_W || mf->get_access_type() == TEXTURE_ACC_R)
    {
        // Data related stuff
        // printf("Data request enters TLB\n");
        // printf("mf - sid:%d, tpc:%d\n", mf->get_sid(), mf->get_tpc());
        m_l1_d_tlb[mf->get_sid()/2]->push_translation_request(time, tr);
        // m_gpu->getMemoryAccessMonitor()->memory_access(mf->get_addr(), mf->is_write());
    }
    else
    {
        // Who are you?
        // assert(0);
        // Send request to complete_queue of apprpriate L1 TLB
        // m_l1_d_tlb[tr->m_mf->get_sid()/2]->push_to_complete_queue(tr);
        m_l1_d_tlb[mf->get_sid()/2]->push_translation_request(time, tr);

    } 
}

void u_tlb_t::push_translation_request(uint64_t time, translation_request* tr)
{
    m_translation_queue.push(tr);
}

void tlb_hierarchy::cycle(uint64_t time)
{
    // L1 TLB cycle
    for (uint64_t i = 0; i < m_l1_d_tlb.size(); i++)
    {
        m_l1_d_tlb[i]->cycle(time);
        /*
        if (m_l1_d_tlb[i]->get_translation_queue_size() != 0)
        {
            printf("L1 TLB [%d] being used\n", i);
            fflush(stdout);
        }
        */
    }

    // L2 TLB cycle
    for (uint64_t j = 0; j < m_l2_u_tlb.size(); j++)
    {
        m_l2_u_tlb[j]->cycle(time);
        //if (m_l2_u_tlb[j]->get_translation_queue_size() != 0)
        //    printf("L2 TLB [%d] being used\n", j);
    }
                 
    // L3 TLB cycle
    m_l3_u_tlb->cycle(time);

    // Page Table Walk
    m_l4_u_tlb->cycle(time);
}

bool tlb_hierarchy::can_commit_translation(tlb_type type, translation_request* tr)
{
    if (type == L1_d_TLB)
        return !m_gpu->icnt_injection_buffer_full(tr->m_mf);
    else return true;
}


void tlb_hierarchy::commit_translation(uint64_t time, tlb_type type,
    translation_request* tr)
{
    switch (type) {

    case L1_d_TLB:
        m_gpu->icnt_inject_request_packet(tr->m_mf);
        delete tr;
        break;
    
    case L2_u_TLB:
        m_l1_d_tlb[tr->m_mf->get_sid()/2]->fill(tr, time);
        tr->set_status(IN_L1TLB_TO_ICNT_QUEUE, time);

        //if (tr->m_mf->get_sid()/2 == 51 && m_l1_d_tlb[51]->has_page_in_tlb(66775061) && time > 520492)
        //{
        //    printf("L2 -> L1 TLB[51]: page_num[%llu], time[%llu]\n", (unsigned long long)tr->addr >> 21, (unsigned long long)time);
        //}


        break;

    case L3_u_TLB:
        m_l2_u_tlb[tr->m_mf->get_sid()/16]->fill(tr, time);
        tr->set_status(IN_L2TLB_TO_L1TLB_QUEUE, time);
        break;
    
    case L4_u_TLB:
        m_l3_u_tlb->fill(tr, time);
        tr->set_status(IN_L3TLB_TO_L2TLB_QUEUE, time);
        break;

    default:
        assert(0);
        break;
    }
}

bool tlb_hierarchy::tlb_miss(uint64_t time, tlb_type type, unsigned tlb_id, translation_request* tr)
{
    switch (type) {
        case L1_i_TLB:
            assert(0);

        case L1_d_TLB: {
            // if (m_l2_u_tlb[l2id]->full()) return false;
            m_l2_u_tlb[tr->m_mf->get_sid()/16]->push_translation_request(time, tr);
            tr->set_status(IN_L2TLB_ACCESS_QUEUE, time);
            break;
        }

        case L2_u_TLB:
            // if (m_l3_u_tlb->full()) return false;
            m_l3_u_tlb->push_translation_request(time, tr);
            tr->set_status(IN_L3TLB_ACCESS_QUEUE, time);
            break;

        case L3_u_TLB: {
            m_l4_u_tlb->push_translation_request(time, tr);
            tr->set_status(IN_L4TLB_ACCESS_QUEUE, time);
            break;
        }

        default:
            assert(0);
    }
    
    return true;
}

void u_tlb_t::print_tlb_info()
{
    printf("complete_queue: %d, miss_queue: %d, merge_queue: %d\n", 
            m_completed_translations.size(), m_pending_missed_translations.size(), m_merged_translations.size());
}

void u_tlb_t::print_tlb_info_detail()
{
    // Print TLB stat
    printf("++ TLB state ++\n");
    for(unsigned i=0; i<m_num_sets; i++)
    {
        printf("Set[%d]: ", i);
        set_t* current_set = m_sets[i];
        current_set->print_tlb_set();
        printf("\n");
    }

    // Print merged_translation queue
    printf("++ merged_translation queue ++\n");
    printf("(Printing first 10 merged requests - Page number)\n");
    printf("> ");
    for (unsigned j=0; j<10; j++)
    {
        translation_request* current_tr = m_merged_translations.front();
        printf("%llu ",(unsigned long long)(current_tr->addr) >> 19);
        m_merged_translations.pop();
    }
    printf("\n");
}

bool u_tlb_t::has_page_in_tlb(uint64_t page_num)
{
    for(unsigned i=0; i<m_num_sets; i++)
    {
        set_t* current_set = m_sets[i];
        if (current_set->check_page_exist(page_num))
            return true;
    }

    return false;
}

void u_tlb_t::print_all_for_debugging(uint64_t time)
{
    // Print time
    printf("== time: %llu ==\n", (unsigned long long)time);
    // Print TLB state (page_num, access_time)
    for(unsigned i=0; i<m_num_sets; i++)
    {
        printf("Set[%d]: ", i);
        set_t* current_set = m_sets[i];
        current_set->print_tlb_entry_time();
    }

    // Print translation queue
    printf("> translation_queue: ");
    std::queue<translation_request*> tmp1 = m_translation_queue;
    unsigned print_size1 = m_translation_queue.size() > 5 ? 5 : m_translation_queue.size();
    for (unsigned j=0; j<print_size1; j++)
    {
        translation_request* current_tr = tmp1.front();
        printf("%llu ",(unsigned long long)(current_tr->addr) >> 19);
        tmp1.pop();
    }
    printf("\n");

    // Print merged queue
    printf("> merged_queue: ");
    std::queue<translation_request*> tmp2 = m_merged_translations;
    unsigned print_size2 = m_merged_translations.size() > 5 ? 5 : m_merged_translations.size();
    for (unsigned j=0; j<print_size2; j++)
    {
        translation_request* current_tr = tmp2.front();
        printf("%llu ",(unsigned long long)(current_tr->addr) >> 19);
        tmp2.pop();
    }
    printf("\n");
}

void u_tlb_t::print_merged_pages_list()
{
    printf("merged_pages: ");
    std::list<uint64_t> tmp = m_merged_pages;
    while (!tmp.empty())
    {
        uint64_t page = tmp.front();
        printf("%llu ", page);
        tmp.pop_front();               // remove the element we just processed
    }
    printf("\n");
    printf("\n");
}

void set_t::entry_flush()
{
    for (unsigned i=0; i<m_entries.size(); i++)
    {
        tlb_entry_t* current_entry = &m_entries[i];
        current_entry->entry_flush();
    }
}

void u_tlb_t::tlb_flush()
{
    for (unsigned i = 0; i < m_num_sets; i++)
        m_sets[i]->entry_flush();
}

void tlb_hierarchy::tlb_flush()
{
    // Flush L1 TLB
    m_n_l1_tlb = 54;
    for (unsigned i = 0; i < m_n_l1_tlb; i++)
        m_l1_d_tlb[i]->tlb_flush();

    // Flush L2 TLB
    m_n_l2_tlb = 7;
    for (unsigned i = 0; i < m_n_l2_tlb; i++)
        m_l2_u_tlb[i]->tlb_flush();

    printf("Flushed TLB!!\n");
}

void tlb_hierarchy::print_tlb_stat()
{
    // Deadlock has occured, SM core is not commiting instructions
    // It would probably be because contention in L2 or L3 TLB cause data dependency not to be satisfied for a long time.
    // So I will loop through TLBs and check at which buffer contention exists.

    // Print L1 TLB
    printf("\n");
    printf("== L1 TLB ==\n");
    for(unsigned i=0; i<m_n_l1_tlb; i++)
    {
        d_tlb_t* current_tlb = m_l1_d_tlb[i];
        printf("TLB[%d]: ", i);
        current_tlb->print_tlb_info();

        if(i == 51)
        {
            printf("L1 TLB[51] id: %d\n", current_tlb->get_tlb_id());
            current_tlb->print_tlb_info_detail();
        }
    }

    // Print L2 TLB
    printf("== L2 TLB ==\n");
    for(unsigned i=0; i<m_n_l2_tlb; i++)
    {
        u_tlb_t* current_tlb = m_l2_u_tlb[i];
        printf("TLB[%d]: ", i);
        current_tlb->print_tlb_info();
    }

    // Print L3 TLB
    printf("== L3 TLB ==\n");
    u_tlb_t* l3_tlb = m_l3_u_tlb;
    l3_tlb->print_tlb_info();

    // Print PTW
    printf("== L4 TLB (PTW) ==\n");
    u_tlb_t* l4_tlb = m_l4_u_tlb;
    l4_tlb->print_tlb_info();
}
